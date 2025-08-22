"""
Calculate the following confusion matrix metrics:
AUC, accuracy, precision, recall, F1-score and Kappa on the WHOLE dataset.
"""
import argparse, os, torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score

from train import get_args_parser
from sloter.slot_model import SlotModel

def infer_one_batch(model, imgs):
    with torch.no_grad():
        out = model(imgs)
    logits = out[0] if isinstance(out, (tuple, list)) else out  # (B, C)
    probs  = torch.softmax(logits, dim=1)                       # (B, C)
    preds  = probs.argmax(dim=1)                                # (B,)
    return probs, preds

def calc_metrics(model, loader, device):  # 모든 배치 누적해 최종 지표 계산
    y_true_all, y_pred_all, pos_prob_all = [], [], []

    model.eval()
    for batch in loader:
        imgs   = batch["image"].to(device, dtype=torch.float32)
        labels = batch["label"].to(device)

        probs, preds = infer_one_batch(model, imgs)

        y_true_all.append(labels.detach().cpu())
        y_pred_all.append(preds.detach().cpu())
        # 이진 분류 가정: class-1 확률
        pos_prob_all.append(probs[:, 1].detach().cpu())

    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()
    pos_pr = torch.cat(pos_prob_all).numpy()

    # 항상 2x2 CM이 되도록 labels=[0,1] 고정
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # AUC는 양/음성 모두 있을 때만 계산 가능
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, pos_pr)
    else:
        auc = float('nan')

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    acc       = (tp + tn) / max(1, (tp + fp + fn + tn))
    kappa     = cohen_kappa_score(y_true, y_pred)

    metrics = {
        "auc":       float(auc),
        "accuracy":  float(acc),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "kappa":     float(kappa),
        "cm":        cm.tolist(),  # 디버깅/기록용
        "counts":    {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "class_present": list(map(int, np.unique(y_true)))
    }
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Metrics", parents=[get_args_parser()])
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="(optional) full path to the .pth checkpoint. "
             "If not set, will look in output_dir for "
             "<dataset>_<model>_<use_slot|no_slot>_checkpoint.pth"
    )
    args = parser.parse_args()

    assert args.dataset in ["ACRIMA", "Blastocyst"], \
        "Only ACRIMA or Blastocyst supported."

    device = torch.device(args.device)

    # 1) 데이터셋/로더 준비
    if args.dataset == "ACRIMA":
        from dataset.ACRIMA import get_data, ACRIMA
        from dataset.transform_func import make_transform
        _, val_data = get_data(args.dataset_dir)
        val_dataset = ACRIMA(val_data, transform=make_transform(args, "val"))
        bs = args.batch_size  # 누적 계산이므로 아무 배치사이즈나 OK
    else:
        from dataset.ConText import ConText, MakeListImage
        from dataset.transform_func import make_transform
        _, _, test_list = MakeListImage(args).get_data()
        val_dataset = ConText(test_list, transform=make_transform(args, "val"))
        bs = args.batch_size

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bs, shuffle=False, num_workers=1, pin_memory=True
    )

    # 2) 체크포인트 로드
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        mode_tag = "use_slot" if args.use_slot else "no_slot"
        ckpt_fname = f"{args.dataset}_{args.model}_{mode_tag}_checkpoint.pth"
        ckpt_path = os.path.join(args.output_dir, ckpt_fname)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint from {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = SlotModel(args).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        print("⚠️  load_state_dict 경고:")
        if missing:
            print("   누락된 파라미터:", missing)
        if unexpected:
            print("   예기치 않은 파라미터:", unexpected)

    # 3) 전체 데이터셋 평가
    metrics = calc_metrics(model, val_loader, device)
    print(metrics)