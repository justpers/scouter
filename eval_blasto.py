from __future__ import print_function
import argparse, os, torch
from PIL import Image
from torchvision import transforms

from train_slot import get_args_parser
from sloter.slot_model import SlotModel
from dataset.ConText import ConText, MakeListImage

from metrics.utils import exp_data
from metrics.IAUC_DAUC import calc_iauc_and_dauc_batch
from metrics.saliency_evaluation.eval_infid_sen import calc_infid_and_sens
from metrics.area_size import calc_area_size

from torch.cuda.amp import autocast

# -------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser('Blastocyst-slot evaluation',
                                parents=[get_args_parser()],
                                conflict_handler='resolve')

    # ── 평가 전용 옵션 ──────────────────────────────────────────────
    p.add_argument('--checkpoint', required=True, help='.pth to evaluate')
    p.add_argument('--auc',      action='store_true')
    p.add_argument('--saliency', action='store_true')
    p.add_argument('--area_prec',action='store_true')

    # ── Blastocyst 기본값 ───────────────────────────────────────────
    p.set_defaults(dataset='Blastocyst',
                   num_classes=2,
                   img_size=260,
                   batch_size=16,
                   use_slot=True,        # 반드시 slot 모델
                   pre_trained=False,
                   vis=True,             # heat-map 자동 저장
                   device='cuda')
    return p


# -------------------------------------------------------------------
def get_val_loader(args):
    train, val, test = MakeListImage(args).get_data()
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    ds = ConText(test, transform=tf)
    return torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=False)


# -------------------------------------------------------------------
def load_model(args, device):
    model = SlotModel(args).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    return model

# -------------------------------------------------------------------
def generate_exps(model, loader, device, loss_status=1, flush_every=32):
    # 저장 폴더 준비
    subdir   = "positive" if loss_status > 0 else "negative"
    save_root = os.path.join("exps", subdir)
    os.makedirs(save_root, exist_ok=True)

    model.eval()
    processed = 0

    with torch.no_grad():
        for batch in loader:
            # 수정: 배치는 CPU에 두고, 이미지 하나씩 GPU로 이동
            imgs, labels, paths = batch["image"], batch["label"], batch["names"]

            for img_cpu, lab, path in zip(imgs, labels, paths):
                # 1) 파일명 처리
                filename = os.path.basename(path)
                base, _  = os.path.splitext(filename)

                # 2) 이미 생성된 heat-map은 건너뜀
                out_path = os.path.join(save_root, base + ".png")
                if os.path.exists(out_path):
                    continue

                # 3) SlotAttention 저장 지시
                if loss_status > 0:
                    save_id = (lab.item(), lab.item(), "exps", base)
                else:
                    lsc     = 1 - lab.item()
                    save_id = (lab.item(), lsc, "exps", base)

                # 이미지 하나만 GPU로 올림
                img = img_cpu.to(device, non_blocking=True)

                # AMP로 메모리/연산량 절감
                with autocast(enabled=torch.cuda.is_available()):
                    out = model(img.unsqueeze(0), save_id=save_id)

                # 임시 텐서/참조 해제
                del out, img
                processed += 1

                # 주기적으로 캐시 비우기
                if torch.cuda.is_available() and (processed % flush_every == 0):
                    torch.cuda.empty_cache()
                    # 선택: 동기화로 누수 의심 시점 정리
                    # torch.cuda.synchronize()
# -------------------------------------------------------------------
def area_size_only(loader, subdir):
    sizes = []
    for batch in loader:
        for name in batch['names']:
            fname = os.path.splitext(os.path.basename(name))[0]
            # 경로 후보: exp_dir 우선, 없으면 sloter/vis 
            cand1 = os.path.join("exps", subdir, f"{fname}.png")
            cand2 = os.path.join('sloter', 'vis', subdir, f"{fname}.png")
            path = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)
            if path is not None:
                sizes.append(calc_area_size(Image.open(path)))
    return sum(sizes)/len(sizes) if sizes else None

# -------------------------------------------------------------------
def main():
    args   = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        
    val_loader = get_val_loader(args)
    model      = load_model(args, device)

    # ── heat-map 먼저 생성 ─────────────────────────────────────────
    if args.auc or args.saliency or args.area_prec:
        print('[Info] generating explanation images …')
        generate_exps(model, val_loader, device, loss_status=args.loss_status, flush_every=32)

    subdir = 'positive' if args.loss_status > 0 else 'negative'
    exp_root = f'exps/{subdir}'
    files  = exp_data.get_exp_filenames(exp_root)

    # ── IAUC / DAUC ────────────────────────────────────────────────
    if args.auc:
        files = exp_data.get_exp_filenames(f'exps/{subdir}')
        exp_loader = torch.utils.data.DataLoader(
            exp_data.ExpData(files, args.img_size, resize=True),
            batch_size=args.batch_size, shuffle=False, num_workers=1)

        iauc, dauc = calc_iauc_and_dauc_batch(
            model, val_loader, exp_loader, args.img_size, device)
        print(f'IAUC={iauc:.4f} | DAUC={dauc:.4f}')

    # ── Infidelity / Sensitivity ───────────────────────────────────
    if args.saliency:
        if args.loss_status < 0:
            lsc_dict = {"0": 1, "1": 0}
        else:
            lsc_dict = {}

        infid_scores, sens_scores = calc_infid_and_sens(
            model, val_loader,
            exp_root,
            loss_status=args.loss_status,
            lsc_dict=lsc_dict)

        # 평균값을 계산해서 출력하거나, 원한다면 dict 전체를 출력해도 됩니다.
        avg_infid = sum(infid_scores.values()) / len(infid_scores)
        avg_sens  = sum(sens_scores.values())  / len(sens_scores)
        print(f'Infidelity={avg_infid:.4f} | Sensitivity={avg_sens:.4f}')

        # 개별 perturbation별 점수를 보고 싶다면:
        print('Infidelity per pert:', infid_scores)
        print('Sensitivity per pert:', sens_scores)

    # ── Area-size (precision 제외) ─────────────────────────────────
    if args.area_prec:
        avg = area_size_only(val_loader, subdir)
        if avg is None:
            print('[Warn] heat-map을 찾지 못했습니다 → area-size 계산 실패')
        else:
            print(f'Average area-size = {avg:.4f}')


# -------------------------------------------------------------------
if __name__ == '__main__':
    main()