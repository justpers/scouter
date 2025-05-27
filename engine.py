import torch
import tools.calculate_tool as cal
from tqdm.auto import tqdm
from collections.abc import Mapping


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

    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        # 딕셔너리 형태 or 튜플 형태 대응
        if isinstance(sample_batch, Mapping):  # dict-like
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
        # total_loss (slot 쓰든 안 쓰든 loss_list[0] 은 항상 총 loss)
        total_loss = loss_list[0]

        if mode == "train":
            total_loss.backward()
            optimizer.step()

        # running loss
        running_loss += total_loss.item()

        # classification (CE) loss: slot 모드이면 loss_list[1], 아니면 total_loss
        if len(loss_list) > 1:
            ce_loss = loss_list[1]
        else:
            ce_loss = total_loss
        running_log_loss += ce_loss.item()

        # attention loss: slot 모드일 때만 loss_list[2], 아니면 0
        if len(loss_list) > 2:
            att_loss = loss_list[2]
        else:
            att_loss = torch.tensor(0.0, device=device)
        running_att_loss += att_loss.item()

        running_corrects += cal.evaluateTop1(logits, labels)

    epoch_loss = round(running_loss / L, 3)
    epoch_loss_log = round(running_log_loss / L, 3)
    epoch_loss_att = round(running_att_loss / L, 3)
    epoch_acc = round(running_corrects / L, 3)
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

