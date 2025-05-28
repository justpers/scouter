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
    # ─── 분산 초기화 ───
    prt.init_distributed_mode(args)
    device = torch.device(args.device)

    # ─── 모델 생성 & 장치 할당 ───
    model = SlotModel(args)
    print("train model:", 
          ("use_slot" if args.use_slot else "no_slot"),
          ("negative_loss" if args.use_slot and args.loss_status != 1 else "positive_loss"))
    model.to(device)
    model_without_ddp = model

    # freeze
    if args.freeze_layers > 0:
        print(f"Freeze first {args.freeze_layers} layers of backbone …")
        model_without_ddp.dfs_freeze(model_without_ddp.backbone, args.freeze_layers)

        # fc만 unfreeze
        if hasattr(model_without_ddp.backbone, 'fc'):
            for p in model_without_ddp.backbone.fc.parameters():
                p.requires_grad = True
    
    if hasattr(model_without_ddp.backbone, 'fc'):
        fc_params = list(model_without_ddp.backbone.fc.parameters())
    else:
        fc_params = []

    # 파라미터 객체 아이디로 필터링 하기 위해 집합 생성
    fc_params_ids = {id(p) for p in fc_params}
    other_params = [
        p for p in model_without_ddp.parameters()
        if p.requires_grad and id(p) not in fc_params_ids
    ]

    optimizer = torch.optim.AdamW([
        {'params': fc_params,     'lr': args.lr_fc},
        {'params': other_params,  'lr': args.lr},
    ], weight_decay=args.weight_decay)

    # ─── Scheduler, Criterion ───
    criterion    = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    # ─── 데이터셋/샘플러 불러오기 ───
    # select_dataset() 는 (train_ds, val_ds, train_sampler) 을 리턴하도록 맞춰 두셔야 합니다
    dataset_train, dataset_val, sampler_train = select_dataset(args)

    if args.distributed:
        train_sampler = sampler_train or DistributedSampler(dataset_train)
        val_sampler   = DistributedSampler(dataset_val, shuffle=False)
    else:
        train_sampler = sampler_train
        val_sampler   = torch.utils.data.SequentialSampler(dataset_val)

    # ─── DataLoader 정의 ───
    data_loader_train = DataLoaderX(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    data_loader_val = DataLoaderX(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ─── 체크포인트 이어받기 ───
    if args.resume:
        cp = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(cp['model'])
        if 'optimizer' in cp and 'lr_scheduler' in cp and 'epoch' in cp:
            optimizer.load_state_dict(cp['optimizer'])
            lr_scheduler.load_state_dict(cp['lr_scheduler'])
            args.start_epoch = cp['epoch'] + 1

    # ─── 학습/평가 루프 ───
    print("Start training")
    start_time = time.time()
    log    = MetricLog()
    record = log.record

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, data_loader_train, optimizer, device, record, epoch)
        lr_scheduler.step()

        # ─── 체크포인트 저장 ───
        if args.output_dir:
            ckpts = []
            base = f"{args.dataset}_" + ("use_slot_" if args.use_slot else "no_slot_")
            if args.use_slot and args.loss_status != 1:
                base += "negative_"
            if args.cal_area_size:
                base += f"for_area_size_{args.lambda_value}_{args.slots_per_class}_"
            ckpts.append(Path(args.output_dir) / (base + "checkpoint.pth"))
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                ckpts.append(Path(args.output_dir) / (base + f"checkpoint{epoch:04}.pth"))
            for p in ckpts:
                prt.save_on_master({
                    'model':       model_without_ddp.state_dict(),
                    'optimizer':   optimizer.state_dict(),
                    'lr_scheduler':lr_scheduler.state_dict(),
                    'epoch':       epoch,
                    'args':        args,
                }, p)

        evaluate(model, data_loader_val, device, record, epoch)
        log.print_metric()

        if "auc" in record["val"]:
            print(f"Epoch {epoch:02d} | Val AUC: {record['val']['auc'][-1]:.4f}")

    total_time = time.time() - start_time
    print('Training time', str(datetime.timedelta(seconds=int(total_time))))
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
