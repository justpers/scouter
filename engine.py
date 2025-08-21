import torch
import tools.calculate_tool as cal
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F


def train_one_epoch(model, data_loader, optimizer, device, record, epoch):
    model.train()
    calculation(model, "train", data_loader, device, record, epoch, optimizer)


@torch.no_grad()
def evaluate(model, data_loader, device, record=None, epoch=0):
    model.eval()

    total_loss = total_ce = total_att = 0.0
    total_correct = total_samples = 0
    all_probs, all_labels = [], []

    for batch in data_loader:
        images = batch["image"].to(device).float()
        labels = batch["label"].to(device)
        bsz = labels.size(0)

        outputs, loss_list = model(images, labels)
        total_ = loss_list[0]
        ce     = loss_list[1] if len(loss_list) >= 2 else F.cross_entropy(outputs, labels)
        att    = loss_list[2] if len(loss_list) >= 3 else 0.0

        preds = outputs.argmax(dim=1)

        total_loss += float(total_.item()) * bsz
        total_ce   += float(ce.item()     if hasattr(ce, "item") else ce) * bsz
        total_att  += float(att.item()    if hasattr(att, "item") else att) * bsz
        total_correct += (preds == labels).sum().item()
        total_samples += bsz

        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)
    avg_ce   = total_ce   / max(1, total_samples)
    avg_att  = total_att  / max(1, total_samples)
    acc      = total_correct / max(1, total_samples)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")

    if record is not None:
        for k in ["loss", "log_loss", "att_loss", "acc", "auc"]:
            if k not in record["val"]:
                record["val"][k] = []
        record["val"]["loss"].append(round(avg_loss, 3))
        record["val"]["log_loss"].append(round(avg_ce, 3))
        record["val"]["att_loss"].append(round(avg_att, 3))
        record["val"]["acc"].append(round(acc, 3))
        record["val"]["auc"].append(round(auc, 4))

    return {"loss": avg_loss, "ce": avg_ce, "att": avg_att, "acc": acc, "auc": auc}

def calculation(model, mode, data_loader, device, record, epoch, optimizer=None):
    from torch.nn import functional as F
    print("start " + mode + " :" + str(epoch))

    total_loss = total_ce = total_att = 0.0
    total_correct = total_samples = 0

    for sample_batch in tqdm(data_loader):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)
        bsz = labels.size(0)

        if mode == "train":
            optimizer.zero_grad()

        outputs, loss_list = model(inputs, labels)
        # total / ce / att 안전 추출
        total = loss_list[0]
        ce    = loss_list[1] if len(loss_list) >= 2 else F.cross_entropy(outputs, labels)
        att   = loss_list[2] if len(loss_list) >= 3 else 0.0

        if mode == "train":
            total.backward()
            optimizer.step()

        # 가중 합(배치 크기만큼 곱해서)
        total_loss += float(total.item()) * bsz
        total_ce   += float(ce.item()   if hasattr(ce, "item") else ce) * bsz
        total_att  += float(att.item()  if hasattr(att, "item") else att) * bsz

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += bsz

    # 배치 크기 가중 평균
    epoch_loss = round(total_loss / max(1, total_samples), 3)
    epoch_ce   = round(total_ce   / max(1, total_samples), 3)
    epoch_att  = round(total_att  / max(1, total_samples), 3)
    epoch_acc  = round(total_correct / max(1, total_samples), 3)

    record[mode]["loss"].append(epoch_loss)
    record[mode]["acc"].append(epoch_acc)
    record[mode]["log_loss"].append(epoch_ce)
    record[mode]["att_loss"].append(epoch_att)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

