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
                   batch_size=8,
                   use_slot=True,        # 반드시 slot 모델
                   pre_trained=False,
                   vis=True,             # ★ heat-map 자동 저장
                   device='cuda')
    return p


# -------------------------------------------------------------------
def get_val_loader(args):
    train, val = MakeListImage(args).get_data()
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    ds = ConText(val, transform=tf)
    return torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=False, num_workers=1, pin_memory=True)


# -------------------------------------------------------------------
def load_model(args, device):
    model = SlotModel(args).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    return model

# -------------------------------------------------------------------
def generate_exps(model, loader, device, loss_status=1):
    # 저장 폴더 준비
    subdir   = "positive" if loss_status > 0 else "negative"
    save_root = os.path.join("exps", subdir)
    os.makedirs(save_root, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs, labels, paths = batch["image"].to(device), batch["label"], batch["names"]
            for img, lab, path in zip(imgs, labels, paths):
                # 1) 원본 파일명에서 확장자만 제거 (e.g. '0054_01.png' → '0054_01')
                filename = os.path.basename(path)
                base, _  = os.path.splitext(filename)

                # 2) 이미 생성된 heat-map은 건너뛰기
                out_path = os.path.join(save_root, base + ".png")
                if os.path.exists(out_path):
                    continue

                # 3) SlotAttention 에게 저장 지시
                # save_id = (GT_class, least_similar_class, root_dir, filename_without_ext)
                lsc = 1 - lab.item()
                _ = model(
                    img.unsqueeze(0),         # (1,3,H,W)
                    save_id=(lab.item(),      # GT class
                             lsc,     
                             "exps",          # 내부에서 subdir을 붙여 exps/positive 또는 exps/negative 로 저장
                             base)            # 파일명 (확장자 제외)
                )
# -------------------------------------------------------------------
def area_size_only(val_loader, subdir):
    sizes = []
    for batch in val_loader:
        fname = os.path.splitext(os.path.basename(batch['names'][0]))[0]
        path = f'sloter/vis/{subdir}/{fname}.png'
        if os.path.exists(path):
            sizes.append(calc_area_size(Image.open(path)))
    return sum(sizes)/len(sizes) if sizes else None

# -------------------------------------------------------------------
def main():
    args   = build_parser().parse_args()
    device = torch.device(args.device)

    val_loader = get_val_loader(args)
    model      = load_model(args, device)

    # ── heat-map 먼저 생성 ─────────────────────────────────────────
    if args.auc or args.saliency or args.area_prec:
        print('[Info] generating explanation images …')
        generate_exps(model, val_loader, device, loss_status=args.loss_status)

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

        # (선택) 개별 perturbation별 점수를 보고 싶다면:
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