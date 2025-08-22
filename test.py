# test.py
from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os
from timm.models import create_model

from sloter.utils.vis import apply_colormap_on_image
from sloter.slot_model import SlotModel
from train import get_args_parser

from dataset.ConText import ConText, MakeList, MakeListImage
from dataset.CUB200 import CUB_200


def test(args, model, device, image_orig, image_tensor, label, vis_id):
    """
    image_orig: PIL.Image (정규화 이전 상태)
    image_tensor: torch.Tensor([C, H, W]) (정규화 후)
    """
    model.to(device)
    model.eval()

    # 1) Forward (slot attention 포함)
    image_tensor = image_tensor.to(device, dtype=torch.float32)
    output = model(image_tensor.unsqueeze(0))  # (1, num_classes)
    pred = output.argmax(dim=1, keepdim=True)
    print("Raw logits:", output[0].detach().cpu().numpy())
    print("Predicted class:", pred.item())

    # 2) 시각화를 위해 원본 이미지를 저장
    os.makedirs('sloter/vis', exist_ok=True)
    image_orig.save('sloter/vis/image.png')

    # 3) 각 슬롯별로 생성된 마스크(slot_{id}.png)를 불러와 heatmap 생성
    for slot_id in range(args.num_classes):
        # slot_{id}.png는 SlotAttention 내부에서 이미 생성되어야 합니다.
        slot_path = f'sloter/vis/slot_{slot_id}.png'
        if not os.path.exists(slot_path):
            print(f"[Warning] {slot_path} 가 존재하지 않습니다. SlotAttention 시각화가 제대로 되었는지 확인하세요.")
            continue

        # PIL로 재불러오기 → numpy 배열로 변환
        slot_mask = np.array(
            Image.open(slot_path).convert('L').resize(image_orig.size, resample=Image.BILINEAR),
            dtype=np.uint8
        )

        # 원본 이미지에도 colormap 적용
        heatmap_only, heatmap_on_image = apply_colormap_on_image(image_orig, slot_mask, 'jet')
        heatmap_only.save(f'sloter/vis/slot_heatmap_{slot_id}.png')
        heatmap_on_image.save(f'sloter/vis/slot_overlay_{slot_id}.png')


def main():
    parser = argparse.ArgumentParser('SCOUTER slot 시각화', parents=[get_args_parser()])
    parser.add_argument('--vis_target_class', type=int, default=0)
    args = parser.parse_args()

    # 1) 인자 타입 고정(convert str→int/float)
    args_dict = vars(args)
    args_for_evaluation = ['num_classes', 'lambda_value', 'power', 'slots_per_class']
    args_type = [int, float, int, int]
    for idx, arg in enumerate(args_for_evaluation):
        args_dict[arg] = args_type[idx](args_dict[arg])

    # 2) 시각화 결과를 저장할 디렉토리
    os.makedirs('sloter/vis', exist_ok=True)

    # 3) checkpoint 파일명 생성 규칙
    checkpoint_path = args.output_dir

    # 4) 장치 설정
    device = torch.device(args.device)

    # 5) “정규화 전” → “ToTensor → Normalize” 순서로 이미지를 불러오기 위한 transform
    base_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    norm_transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    if args.dataset == 'Blastocyst':
    # 🔸 시각화하고 싶은 클래스 지정 (예: class 1)
        vis_target_class = 1

        # Blastocyst는 MakeListImage → ConText(index→(path,label)) 방식 사용
        train_list, val_list, test_list = MakeListImage(args).get_data()
        dataset_val = ConText(train_list, transform=base_transform)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.batch_size,
            shuffle=False, num_workers=1, pin_memory=True
        )

        found = False
        for batch in data_loader_val:
            for i in range(len(batch["label"])):
                label_i = batch["label"][i].item()
                if label_i == vis_target_class:
                    image_tensor = batch["image"][i]  # Normalize 전
                    label = label_i
                    image_orig = Image.fromarray(
                        (image_tensor.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8),
                        mode='RGB'
                    )
                    image_tensor = norm_transform(image_tensor)
                    found = True
                    break
            if found:
                break

        if not found:
            raise ValueError(f"💥 클래스 {vis_target_class} 이미지를 검증 데이터셋에서 찾을 수 없습니다.")

    elif args.dataset == 'ImageNet':
        # ImageNet 역시 MakeListImage → ConText 방식
        train_list, val_list = MakeListImage(args).get_data()
        dataset_val = ConText(val_list, transform=base_transform)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.batch_size,
            shuffle=False, num_workers=1, pin_memory=True
        )
        batch = next(iter(data_loader_val))
        image_tensor = batch["image"][0]
        label = batch["label"][0].item()
        image_orig = Image.fromarray(
            (image_tensor.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8),
            mode='RGB'
        )
        image_tensor = norm_transform(image_tensor)

    elif args.dataset == 'ConText':
        # ConText(Chest X-ray)도 MakeList → ConText 사용
        train_list, val_list = MakeList(args).get_data()
        dataset_val = ConText(val_list, transform=base_transform)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.batch_size,
            shuffle=False, num_workers=1, pin_memory=True
        )
        batch = next(iter(data_loader_val))
        image_tensor = batch["image"][0]
        label = batch["label"][0].item()
        image_orig = Image.fromarray(
            (image_tensor.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8),
            mode='RGB'
        )
        # ConText / Blastocyst / ImageNet 모두 동일한 normalization 사용
        image_tensor = norm_transform(image_tensor)

    elif args.dataset == 'MNIST':
        # MNIST: torchvision.datasets 사용
        # (grayscale → 1채널 텐서)
        dataset_val = datasets.MNIST(
            './data/mnist', train=False,
            transform=transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.batch_size,
            shuffle=False, num_workers=1, pin_memory=True
        )
        image_tensor, _ = next(iter(data_loader_val))
        image_tensor = image_tensor[0]         # (1, H, W)
        label = 'N/A'                         # MNIST에선 slot attention 시각화 의미 없음
        image_orig = Image.fromarray(
            (image_tensor.cpu().numpy()[0] * 255).astype(np.uint8),
            mode='L'
        )
        # MNIST normalization은 이미 transform 내부에 적용됨

    elif args.dataset == 'CUB200':
        # CUB-200: custom CUB_200 클래스 사용
        dataset_val = CUB_200(args, train=False, transform=base_transform)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.batch_size,
            shuffle=False, num_workers=1, pin_memory=True
        )
        batch = next(iter(data_loader_val))
        image_tensor = batch["image"][0]
        label = batch["label"][0].item()
        image_orig = Image.fromarray(
            (image_tensor.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8),
            mode='RGB'
        )
        image_tensor = norm_transform(image_tensor)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print("Ground-truth label:\t", label)

    # 7) 모델 초기화 및 체크포인트 로드
    model = SlotModel(args)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    print("Checkpoint keys:", list(checkpoint.keys()))
    model.load_state_dict(checkpoint["model"], strict=True)

    # 8) 실제 테스트/시각화 호출
    test(args, model, device, image_orig, image_tensor, label, vis_id=args.vis_id)


if __name__ == '__main__':
    main()
