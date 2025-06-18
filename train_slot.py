import argparse
from pathlib import Path
import torch
from torch.utils.data import DistributedSampler
import tools.prepare_things as prt
from engine import train_one_epoch, evaluate
from dataset.choose_dataset import select_dataset
#from tools.prepare_things import DataLoaderX
from torch.utils.data import DataLoader
from sloter.slot_model import SlotModel
from tools.calculate_tool import MetricLog
import datetime
import time
import numpy as np
from thop import profile, clever_format
import tensorly as tl
import gc
from sklearn.metrics import roc_auc_score
import random


def get_args_parser():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser('Set SCOUTER model', add_help=False)
    parser.add_argument('--model', default="resnet18", type=str)
    parser.add_argument('--dataset', default="MNIST", type=str)
    parser.add_argument('--channel', default=512, type=int)

    # training set
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_drop', default=70, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument('--img_size', default=260, help='path for save data')
    parser.add_argument('--pre_trained', default=True, type=str2bool, help='whether use pre parameter for backbone')
    parser.add_argument('--use_slot', default=True, type=str2bool, help='whether use slot module')
    parser.add_argument('--use_pre', default=False, type=str2bool, help='whether use pre dataset parameter')
    parser.add_argument('--aug', default=False, type=str2bool, help='whether use pre dataset parameter')
    parser.add_argument('--grad', default=False, type=str2bool, help='whether use grad-cam for visulazition')
    parser.add_argument('--grad_min_level', default=0., type=float, help='control the grad-cam vis area')
    parser.add_argument('--iterated_evaluation_num', default=1, type=int, help='used for iterated evaluation')
    parser.add_argument('--cal_area_size', default=False, type=str2bool, help='whether to calculate for area size of the attention map')
    parser.add_argument('--thop', default=False, type=str2bool, help='whether to only calculate for the model costs (no training)')

    # slot setting
    parser.add_argument('--loss_status', default=1, type=int, help='positive or negative loss')
    parser.add_argument('--freeze_layers', default=2, type=int, help='number of freeze layers')
    parser.add_argument('--hidden_dim', default=64, type=int, help='dimension of to_k')
    parser.add_argument('--slots_per_class', default=1, type=int, help='number of slot for each class')
    parser.add_argument('--power', default=2, type=int, help='power of the slot loss')
    parser.add_argument('--to_k_layer', default=1, type=int, help='number of layers in to_k')
    parser.add_argument('--lambda_value', default=1, type=float, help='lambda of slot loss')
    parser.add_argument('--vis', default=False, type=str2bool, help='whether save slot visualization')
    parser.add_argument('--vis_id', default=0, type=int, help='choose image to visualization')

    # data/machine set
    parser.add_argument('--dataset_dir', default='../PAN/bird_200/CUB_200_2011/CUB_200_2011/',
                        help='path for save data')
    parser.add_argument('--output_dir', default='saved_model/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--pre_dir', default='pre_model/',
                        help='path of pre-train model')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--resume', default=None, type=str, help='경로 입력 시 해당 체크포인트 불러와 학습')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 1) DDP 초기화 (필요하다면)
    prt.init_distributed_mode(args)
    device = torch.device(args.device)

    # 3) 모델 생성
    model = SlotModel(args).to(device)
    model_without_ddp = model if not args.distributed else model.module

    # 4) Optimizer / Criterion / LR Scheduler
    params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # Label-smoothing 적용
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # ReduceLROnPlateau: val_loss 기준, 3 epoch 개선 없으면 LR × 0.5
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # 5) 데이터셋 및 DataLoader
    dataset_train, dataset_val = select_dataset(args)
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val   = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val   = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler_train,
        num_workers=args.num_workers,
        drop_last=True
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        num_workers=args.num_workers
    )

    # 6) Resume 체크포인트 (옵션)
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(ckpt['model'], strict=False)
        start_epoch = 0
        best_acc = 0.0
        print(f"Resume from {args.resume} → backbone만 로드, 학습은 처음부터 이어서 진행합니다.")
        
    # 7) 로그 초기화
    log = MetricLog()
    record = log.record

    # 8) output 디렉토리
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_epoch = start_epoch
    print(f"Start training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    tic = time.time()

    # 9) Epoch 루프
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # (A) 학습 1 에폭
        train_one_epoch(model, data_loader_train, optimizer, device, record, epoch)

        # (B) 검증 및 성능 계산
        val_res = evaluate(model, data_loader_val, device, record, epoch)
        val_loss = val_res['loss']
        val_acc = val_res['acc']
        val_auc = val_res['auc']
        print(f"[Epoch {epoch}] val_acc={val_acc:.3f} val_auc={val_auc:.3f} val_loss={val_loss:.3f}")

        # (C) LR 스케줄링 (val_loss 기준)
        lr_scheduler.step(val_loss)

        # (D) best.pth 저장 (val_acc 기준)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            save_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'best_acc': best_acc
            }
            torch.save(save_dict, output_dir/'best.pth')
            print("✔ best.pth updated")

        # (E) Early-Stopping (patience: val_acc 개선 없을 시 중단)
        if epoch - best_epoch >= args.patience:
            print(f"Early-Stopping at epoch {epoch} (no val_acc ↑ for {args.patience} epochs)")
            break

        # (F) 주기적 체크포인트 (예: 10 에폭마다 저장)
        if (epoch + 1) % 10 == 0:
            ckpt_path = output_dir / f'checkpoint_{epoch:04}.pth'
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'best_acc': best_acc
            }, ckpt_path)
            print(f"→ saved checkpoint: {ckpt_path.name}")

        # (G) 메모리 정리 및 로그 출력
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        log.print_metric()

    total_time = time.time() - tic
    print("Total training time:", str(datetime.timedelta(seconds=int(total_time))))
# ---------------------------------------------------------------
# 3. Entrypoint
# ---------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    cfg = parser.parse_args()
    main(cfg)