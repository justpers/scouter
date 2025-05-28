import argparse
from pathlib import Path
import torch
from torch.utils.data import DistributedSampler
import tools.prepare_things as prt
from engine import train_one_epoch, evaluate
from dataset.choose_dataset import select_dataset
from tools.prepare_things import DataLoaderX
from sloter.slot_model import SlotModel
from tools.calculate_tool import MetricLog
import datetime
import time


def get_args_parser():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser('Set SCOUTER model', add_help=False)
    # ─── 기본 ───
    parser.add_argument('--model', default="resnet18", type=str)
    parser.add_argument('--dataset', default="MNIST", type=str)
    parser.add_argument('--img_size', default=260, type=int, help='input image size for Resize()')
    parser.add_argument('--dataset_dir', default='./data/', help='path for data')
    parser.add_argument('--output_dir',  default='saved_model/', help='where to save checkpoints')
    parser.add_argument('--device',      default='cuda', help='device')
    parser.add_argument('--num_workers', default=2,    type=int)
    # ─── 학습 하이퍼파라미터 ───
    parser.add_argument('--lr',       default=1e-4,  type=float, help='learning rate for backbone')
    parser.add_argument('--lr_fc',    default=1e-3,  type=float, help='learning rate for fc/slot module')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size',   default=64,   type=int)
    parser.add_argument('--epochs',       default=20,   type=int)
    parser.add_argument('--lr_drop',      default=70,   type=int, help='stepLR drop interval')
    parser.add_argument('--early_stop_patience', default=5, type=int, help='val AUC 개선이 없으면 몇 epoch 뒤 중단할지')
    parser.add_argument('--best_ckpt_name', default='best_auc.pth', type=str, help='AUC 최고 모델 저장 파일명')
    # ─── SCOUTER 옵션 ───
    parser.add_argument('--num_classes', default="10", type=str)
    parser.add_argument('--use_slot',     default=True, type=str2bool)
    parser.add_argument('--freeze_layers', default=0,    type=int,
                        help='freeze first N layers of backbone')
    parser.add_argument('--lambda_value',  default="1.", type=str)
    parser.add_argument('--slots_per_class', default="3", type=str)
    parser.add_argument('--power',           default="2", type=str)
    parser.add_argument('--to_k_layer',      default=1,   type=int)
    parser.add_argument('--loss_status',     default=1,   type=int)
    parser.add_argument('--hidden_dim',      default=64,  type=int)
    parser.add_argument('--channel',         default=512, type=int)
    parser.add_argument('--use_pre',  default=False, type=str2bool)
    parser.add_argument('--pre_trained', default=True, type=str2bool)
    parser.add_argument('--vis',      default=False, type=str2bool)
    parser.add_argument('--vis_id',   default=0,     type=int)
    # ─── 기타 기능 ───
    parser.add_argument('--aug',    default=True,  type=str2bool, help='enable data augmentation')
    parser.add_argument('--aug_level', default='base', type=str, choices=['base', 'strong'], help='augmentation 강도(base/strong)')
    parser.add_argument('--grad',   default=False, type=str2bool)
    parser.add_argument('--grad_min_level', default=0., type=float)
    parser.add_argument('--cal_area_size', default=False, type=str2bool)
    parser.add_argument('--iterated_evaluation_num', default=1, type=int)
    parser.add_argument('--thop',   default=False, type=str2bool)
    # ─── 분산 학습 ───
    parser.add_argument('--world_size',  default=1, type=int)
    parser.add_argument('--local_rank',  type=int)
    parser.add_argument('--dist_url',    default='env://', help='distributed URL')
    parser.add_argument('--resume',      default=False, type=str2bool)
    parser.add_argument('--start_epoch', default=0,     type=int)
    return parser


def main(args):
    # ─── 분산/장치 초기화 ────────────────────────────────────────────────
    prt.init_distributed_mode(args)
    device = torch.device(args.device)

    # ─── 모델 생성 ─────────────────────────────────────────────────────
    model = SlotModel(args)
    print("train model:",
          "use slot" if args.use_slot else "without slot",
          "negative loss" if args.use_slot and args.loss_status != 1 else "positive loss")
    model.to(device)
    model_without_ddp = model

    # ─── DDP 설정 ─────────────────────────────────────────────────────
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # ─── Optimizer 분리 설정 (backbone vs fc/slot) ───────────────────
    fc_params = []
    backbone_params = []

    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone.fc' in name or 'slot' in name or 'conv1x1' in name:
            fc_params.append(param)
        else:
            backbone_params.append(param)

    # 최종 Optimizer 구성
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr},
        {'params': fc_params, 'lr': args.lr_fc}
    ], weight_decay=args.weight_decay)

    criterion     = torch.nn.CrossEntropyLoss()
    lr_scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    # ─── 데이터셋 & DataLoader ─────────────────────────────────────────
    dataset_train, dataset_val, sampler_train = select_dataset(args)
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val   = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val   = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoaderX(dataset_train, batch_sampler=batch_sampler_train,
                                    num_workers=args.num_workers)
    data_loader_val   = DataLoaderX(dataset_val, args.batch_size, sampler=sampler_val,
                                    num_workers=args.num_workers)

    # ─── 체크포인트 이어받기 ───────────────────────────────────────────
    output_dir = Path(args.output_dir)
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(ckpt['model'])
        if all(k in ckpt for k in ['optimizer', 'lr_scheduler', 'epoch']):
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            args.start_epoch = ckpt['epoch'] + 1
            print(f"▶ Resumed from epoch {ckpt['epoch']} (AUC={ckpt.get('auc', 'N/A')})")

    # ─── 로그 & Early-Stopping 변수 ────────────────────────────────────
    print("Start training")
    start_time   = time.time()
    log          = MetricLog()
    record       = log.record
    best_auc     = -1.0
    patience_cnt = 0

    # ─── Epoch 루프 ────────────────────────────────────────────────────
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_one_epoch(model, data_loader_train, optimizer, device, record, epoch)
        lr_scheduler.step()

        # ─── 주기적 체크포인트(전체 모델) ─────────────────────────────
        if args.output_dir:
            base = f"{args.dataset}_{'use_slot_' if args.use_slot else 'no_slot_'}" + \
                   (f"negative_" if args.use_slot and args.loss_status != 1 else "")
            if args.cal_area_size:
                base += f"for_area_size_{args.lambda_value}_{args.slots_per_class}_"
            cp_paths = [output_dir / (base + "checkpoint.pth")]
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                cp_paths.append(output_dir / (base + f"checkpoint{epoch:04}.pth"))
            for p in cp_paths:
                prt.save_on_master({
                    'model':        model_without_ddp.state_dict(),
                    'optimizer':    optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch':        epoch,
                    'args':         args,
                }, p)

        # ─── Validation & 로그 출력 ───────────────────────────────────
        evaluate(model, data_loader_val, device, record, epoch)
        log.print_metric()

        cur_auc = record['val'].get('auc', [None])[-1]
        if cur_auc is not None:
            print(f"Epoch {epoch:02d} | "
                  f"Train Acc {record['train']['acc'][-1]:.3f} | "
                  f"Val Acc {record['val']['acc'][-1]:.3f} | "
                  f"Val AUC {cur_auc:.4f}")

            # ── Best-AUC 체크포인트 ───────────────────────────────
            if cur_auc > best_auc:
                best_auc = cur_auc
                patience_cnt = 0
                best_path = output_dir / args.best_ckpt_name
                prt.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'auc': best_auc,
                    'args': args,
                }, best_path)
                print(f"  ▲ New best AUC! checkpoint saved to {best_path.name}")
            else:
                patience_cnt += 1
                print(f"  ▼ AUC not improved ({patience_cnt}/{args.early_stop_patience})")

            # ── Early Stopping ──────────────────────────────────
            if patience_cnt >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best Val AUC = {best_auc:.4f}")
                break

    # ─── 훈련 종료 로그 ───────────────────────────────────────────────
    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {elapsed}")
    return [record["train"]["acc"][-1], record["val"]["acc"][-1]]


def param_translation(args):
    args_dict = vars(args)
    # 반복 실험할 인자들
    grid_args = ['num_classes', 'lambda_value', 'power', 'slots_per_class', 'hidden_dim']
    grid_types= [int, float,     int,      int,               int     ]
    target_arg = None

    # 쉼표로 구분된 반복 설정 감지
    for i, key in enumerate(grid_args):
        v = args_dict[key]
        if isinstance(v, str) and ',' in v:
            target_arg  = key
            target_type = grid_types[i]
            grid_vals   = v.split(",")
            break
        else:
            args_dict[key] = grid_types[i](v)

    if target_arg is None:
        main(args)
    else:
        results = {}
        for val in grid_vals:
            args_dict[target_arg] = target_type(val)
            results[f"{target_arg}-{val}"] = [ main(args) for _ in range(args.iterated_evaluation_num) ]
            print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    param_translation(args)
