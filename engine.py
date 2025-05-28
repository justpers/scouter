import torch
import tools.calculate_tool as cal
from tqdm.auto import tqdm
from collections.abc import Mapping
from sklearn.metrics import roc_auc_score  # ← 추가

def train_one_epoch(model, data_loader, optimizer, device, record, epoch):
    model.train()
    calculation(model, "train", data_loader, device, record, epoch, optimizer)


@torch.no_grad()
def evaluate(model, data_loader, device, record, epoch):
    model.eval()
    calculation(model, "val", data_loader, device, record, epoch)


def calculation(model, mode, data_loader, device, record, epoch, optimizer=None):
    L = len(data_loader)
    running_loss = 0.0
    running_corrects = 0.0
    running_att_loss = 0.0
    running_log_loss = 0.0
    print("start " + mode + " :" + str(epoch))

    all_labels = []  # ← AUC용
    all_probs = []

    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        # 딕셔너리 형태 or 튜플 형태 대응
        if isinstance(sample_batch, Mapping):
            inputs = sample_batch["image"].to(device, dtype=torch.float32)
            labels = sample_batch["label"].to(device, dtype=torch.int64)
        elif isinstance(sample_batch, (list, tuple)) and len(sample_batch) == 2:
            inputs, labels = sample_batch
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.int64)
        else:
            raise TypeError(f"Unsupported sample_batch format: {type(sample_batch)}")

        if mode == "train":
            optimizer.zero_grad()

        logits, loss_list = model(inputs, labels)
        total_loss = loss_list[0]

        if mode == "train":
            total_loss.backward()
            optimizer.step()

        running_loss += total_loss.item()
        ce_loss = loss_list[1] if len(loss_list) > 1 else total_loss
        running_log_loss += ce_loss.item()
        att_loss = loss_list[2] if len(loss_list) > 2 else torch.tensor(0.0, device=device)
        running_att_loss += att_loss.item()
        running_corrects += cal.evaluateTop1(logits, labels)

        # AUC 계산을 위한 확률 및 라벨 저장
        if mode == "val":
            probs = torch.exp(logits)  # log_softmax → softmax
            all_probs.extend(probs[:, 1].detach().cpu().numpy())  # 양성 클래스 확률
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = round(running_loss / L, 3)
    epoch_loss_log = round(running_log_loss / L, 3)
    epoch_loss_att = round(running_att_loss / L, 3)
    epoch_acc = round(running_corrects / L, 3)

    record[mode]["loss"].append(epoch_loss)
    record[mode]["acc"].append(epoch_acc)
    record[mode]["log_loss"].append(epoch_loss_log)
    record[mode]["att_loss"].append(epoch_loss_att)

    # ROC-AUC 기록
    if mode == "val":
        try:
            auc_score = roc_auc_score(all_labels, all_probs)
        except:
            auc_score = 0.5  # 실패 시 기본값
        record[mode].setdefault("auc", []).append(round(auc_score, 4))

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

