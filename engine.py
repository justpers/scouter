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

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_log = 0.0       # ← CE loss 누적
    total_att = 0.0       # ← attention loss 누적
    all_probs, all_labels = [], []

    for batch in data_loader:
        images = batch["image"].to(device).float()
        labels = batch["label"].to(device)

        # 모델 출력 및 loss_list 가져오기
        outputs, loss_list = model(images, labels)
        loss = loss_list[0]
        ce   = loss_list[1]
        att  = loss_list[2] if len(loss_list) > 2 else torch.tensor(0.0, device=device)

        preds = outputs.argmax(dim=1)

        total_loss    += loss.item() * images.size(0)
        total_log     += ce.item()   * images.size(0)
        total_att     += att.item()  * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        probs = torch.softmax(outputs, dim=1)[:, 1]  # binary 분류 가정
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_ce   = total_log  / total_samples
    avg_att  = total_att  / total_samples
    acc      = total_correct / total_samples
    auc      = roc_auc_score(all_labels, all_probs)

    if record is not None:
        # 기존 train_one_epoch() 에서 사용하는 "log_loss" / "att_loss" 키와 동일하게 사용해야 저장됨
        for k in ["loss", "log_loss", "att_loss", "acc", "auc"]:
            if k not in record["val"]:
                record["val"][k] = []
        record["val"]["loss"].append(round(avg_loss, 3))
        record["val"]["log_loss"].append(round(avg_ce, 3))   # CE 손실
        record["val"]["att_loss"].append(round(avg_att, 3))  # attention 손실
        record["val"]["acc"].append(round(acc, 3))
        record["val"]["auc"].append(round(auc, 4))

    return {"loss": avg_loss, "ce": avg_ce, "att": avg_att, "acc": acc, "auc": auc}

def calculation(model, mode, data_loader, device, record, epoch, optimizer=None):
    L = len(data_loader)
    running_loss = 0.0
    running_corrects = 0.0
    running_att_loss = 0.0
    running_log_loss = 0.0
    print("start " + mode + " :" + str(epoch))
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)

        if mode == "train":
            optimizer.zero_grad()
        logits, loss_list = model(inputs, labels)
        loss = loss_list[0]
        if mode == "train":
            loss.backward()
            # clip_gradient(optimizer, 1.1)
            optimizer.step()

        a = loss.item()
        running_loss += a
        if len(loss_list) > 2: # For slot training only
            running_att_loss += loss_list[2].item()
            running_log_loss += loss_list[1].item()
        running_corrects += cal.evaluateTop1(logits, labels)
        # if i_batch % 10 == 0:
        #     print("epoch: {} {}/{} Loss: {:.4f}".format(epoch, i_batch, L-1, a))
    epoch_loss = round(running_loss/L, 3)
    epoch_loss_log = round(running_log_loss/L, 3)
    epoch_loss_att = round(running_att_loss/L, 3)
    epoch_acc = round(running_corrects/L, 3)
    record[mode]["loss"].append(epoch_loss)
    record[mode]["acc"].append(epoch_acc)
    record[mode]["log_loss"].append(epoch_loss_log)
    record[mode]["att_loss"].append(epoch_loss_att)


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