# from __future__ import print_function
# import argparse
# import torch
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from PIL import Image
# import numpy as np
# from timm.models import create_model
# import os, os.path
# from sloter.utils.vis import apply_colormap_on_image
# from sloter.slot_model import SlotModel
# from train import get_args_parser

# from torchvision import datasets, transforms
# from dataset.ConText import ConText, MakeList, MakeListImage
# from dataset.CUB200 import CUB_200

# def test(args, model, device, img, image, label, vis_id):
#     model.to(device)
#     model.eval()
#     image = image.to(device, dtype=torch.float32)
#     output = model(torch.unsqueeze(image, dim=0))
#     pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#     print(output[0])
#     print(pred[0])

#     #For vis
#     image_raw = img
#     image_raw.save('sloter/vis/image.png')
#     print(torch.argmax(output[vis_id]).item())
#     model.train()

#     for id in range(args.num_classes):
#         image_raw = Image.open('sloter/vis/image.png').convert('RGB')
#         slot_image = np.array(Image.open(f'sloter/vis/slot_{id}.png').resize(image_raw.size, resample=Image.BILINEAR), dtype=np.uint8)

#         heatmap_only, heatmap_on_image = apply_colormap_on_image(image_raw, slot_image, 'jet')
#         heatmap_on_image.save(f'sloter/vis/slot_mask_{id}.png')

#     if args.cal_area_size:
#         slot_image = np.array(Image.open(f'sloter/vis/slot_{str(label) if args.loss_status>0 else str(label+1)}.png'), dtype=np.uint8)
#         slot_image_size = slot_image.shape
#         attention_ratio = float(slot_image.sum()) / float(slot_image_size[0]*slot_image_size[1]*255)
#         print(f"attention_ratio: {attention_ratio}")


# def main():
#     parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
#     args = parser.parse_args()

#     args_dict = vars(args)
#     args_for_evaluation = ['num_classes', 'lambda_value', 'power', 'slots_per_class']
#     args_type = [int, float, int, int]
#     for arg_id, arg in enumerate(args_for_evaluation):
#         args_dict[arg] = args_type[arg_id](args_dict[arg])

#     os.makedirs('sloter/vis', exist_ok=True)

#     model_name = f"{args.dataset}_" + f"{'use_slot_' if args.use_slot else 'no_slot_'}"\
#                 + f"{'negative_' if args.use_slot and args.loss_status != 1 else ''}"\
#                 + f"{'for_area_size_'+str(args.lambda_value) + '_'+ str(args.slots_per_class) + '_' if args.cal_area_size else ''}" + 'checkpoint.pth'
#     args.use_pre = False

#     device = torch.device(args.device)
    
#     transform = transforms.Compose([
#         transforms.Resize((args.img_size, args.img_size)),
#         transforms.ToTensor(),
#         ])
#     # Con-text
#     if args.dataset == 'ConText':
#         train, val = MakeList(args).get_data()
#         dataset_val = ConText(val, transform=transform)
#         data_loader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
#         data = iter(data_loader_val).next()
#         image = data["image"][0]
#         label = data["label"][0].item()
#         image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8).transpose((1,2,0)), mode='RGB')
#         image = transform(image_orl)
#         transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     elif args.dataset == 'ImageNet':
#         train, val = MakeListImage(args).get_data()
#         dataset_val = ConText(val, transform=transform)
#         data_loader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
#         iter_loader = iter(data_loader_val)
#         for i in range(0, 1):
#             data = iter_loader.next()
#         image = data["image"][0]
#         label = data["label"][0].item()
#         image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8).transpose((1,2,0)), mode='RGB')
#         image = transform(image_orl)
#         transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     # MNIST
#     elif args.dataset == 'MNIST':
#         dataset_val = datasets.MNIST('./data/mnist', train=False, transform=transform)
#         data_loader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
#         image = next(iter(data_loader_val))[0][0]
#         label = ''
#         image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8)[0], mode='L')
#         image = transform(image_orl)
#         transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
#     # CUB
#     elif args.dataset == 'CUB200':
#         dataset_val = CUB_200(args, train=False, transform=transform)
#         data_loader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
#         data = iter(data_loader_val).next()
#         image = data["image"][0]
#         label = data["label"][0].item()
#         image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8).transpose((1,2,0)), mode='RGB')
#         image = transform(image_orl)
#         transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     image = transform(image)

#     print("label\t", label)
#     model = SlotModel(args)
#     # Map model to be loaded to specified single gpu.
#     checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
#     for k, v in checkpoint.items():
#         print(k)
#     model.load_state_dict(checkpoint["model"])

#     test(args, model, device, image_orl, image, label, vis_id=args.vis_id)


# if __name__ == '__main__':
#     main()
# test_blasto.py
from __future__ import print_function
import argparse, random, os, json, time
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image

from train import get_args_parser
from sloter.slot_model import SlotModel
from sloter.utils.vis import apply_colormap_on_image

# ── util : reproducibility ─────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ── CLI ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("SCOUTER blastocyst inference", parents=[get_args_parser()])
parser.add_argument("--ckpt", required=True, help="fine-tuned checkpoint (.pth)")
parser.add_argument("--vis_out", default="vis_blasto")
parser.add_argument("--idx", default=0, type=int)
args = parser.parse_args()

# ★ 숫자형 인자 강제 캐스팅 --------------------------------------------
for k, tp in [("num_classes", int),
              ("slots_per_class", int),
              ("hidden_dim", int),
              ("power", int),
              ("lambda_value", float)]:
    setattr(args, k, tp(getattr(args, k)))

set_seed(777)

# ── 기본 transform (train.py 와 동일) ─────────────────────────────────
tf_resize = transforms.Resize((args.img_size, args.img_size))
tf_toT    = transforms.ToTensor()
tf_norm   = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

# ★ blastocyst 데이터셋 로드 (ImageFolder) ------------------------------
val_dir   = Path(args.dataset_dir)/"val"
ds_val    = datasets.ImageFolder(val_dir, transform=tf_toT)
assert len(ds_val)>0, f"val 폴더가 비어있습니다: {val_dir}"
img_raw, label = ds_val[args.idx]
print(f"sample index {args.idx}  |  label={label}  |  path={ds_val.samples[args.idx][0]}")

# raw PIL 이미지 확보
pil_raw = Image.open(ds_val.samples[args.idx][0]).convert("RGB")
pil_res = tf_resize(pil_raw)
tensor_in= tf_norm(tf_toT(pil_res))          # (3,H,W), float

# ── 모델 로드 ──────────────────────────────────────────────────────────
device  = torch.device(args.device)
model   = SlotModel(args).to(device)
ckpt    = torch.load(args.ckpt, map_location="cpu")
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
print(f"✔ checkpoint loaded : AUC={ckpt.get('auc','?')}  epoch={ckpt.get('epoch','?')}")

# ── 추론 ───────────────────────────────────────────────────────────────
with torch.no_grad():
    logit = model(tensor_in.unsqueeze(0).to(device))   # (1,2)
    prob  = torch.softmax(logit,1)[0,1].item()
pred = int(torch.argmax(logit,1))
print(f"pred={pred}  prob(positive)={prob:.3f}")

# ── heatmap 수집 & 저장 ────────────────────────────────────────────────
os.makedirs(args.vis_out, exist_ok=True)
save_image(tensor_in, f"{args.vis_out}/input.png")

if hasattr(model.slot, "vis_map"):
    pos_map, neg_map = model.slot.vis_map  # (1,1,H,W)
    for tag, m in zip(["pos","neg"], [pos_map[0], neg_map[0]]):
        m  = m.cpu() / m.max().clamp(min=1e-8)
        rgb= torch.cat([m, torch.zeros_like(m), 1-m],0)
        save_image(rgb, f"{args.vis_out}/{tag}_heat.png")

        # overlay
        img_np = np.array(pil_res)
        mask   = (rgb.permute(1,2,0).numpy()*255).astype(np.uint8)
        hm_only, hm_overlay = apply_colormap_on_image(Image.fromarray(img_np), mask, 'jet')
        hm_overlay.save(f"{args.vis_out}/{tag}_overlay.png")
else:
    print("⚠️  model.slot.vis_map 가 존재하지 않아 heatmap 저장이 스킵되었습니다.")

# ── metrics 기록 ──────────────────────────────────────────────────────
json.dump({"pred": pred, "prob": prob}, open(f"{args.vis_out}/meta.json","w"), indent=2)
print(f"saved to {args.vis_out}/")
