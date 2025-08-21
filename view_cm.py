# view_cm.py – 체크포인트와 데이터를 받아 confusion matrix만 보여 주는 예시
import argparse, torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sloter.slot_model import SlotModel
from dataset.ConText import ConText, MakeListImage
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', required=True)
parser.add_argument('--checkpoint',  required=True)
parser.add_argument('--model',       default='resnest26d')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--img_size',    type=int, default=260)
parser.add_argument('--batch_size',  type=int, default=32)
parser.add_argument('--dataset', default="MNIST", type=str)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 데이터
_, val = MakeListImage(args).get_data()
tf = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
loader = torch.utils.data.DataLoader(
    ConText(val, transform=tf), batch_size=32, shuffle=False)

# 모델
ckpt = torch.load(args.checkpoint, map_location=device)

slot_args = ckpt["args"]          # ← 훈련 때 쓰던 모든 하이퍼파라미터 포함
slot_args.pre_trained = False     # 평가 시엔 필요없지만, 있어도 무방
slot_args.device = device         # 혹시 없다면 추가

model = SlotModel(slot_args).to(device)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

# 예측
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in loader:
        x = batch['image'].to(device)
        y = batch['label'].to(device)
        out = model(x)
        probs = out[0] if isinstance(out,(tuple,list)) else out
        all_preds.append(torch.argmax(probs,1).cpu())
        all_labels.append(y.cpu())
cm = confusion_matrix(torch.cat(all_labels), torch.cat(all_preds))

# 그림
fig, ax = plt.subplots()
im = ax.imshow(cm)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha='center', va='center', color='w')
plt.title("Confusion Matrix")
plt.tight_layout(); plt.show()