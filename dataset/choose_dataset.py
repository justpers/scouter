from torchvision.datasets import MNIST
from torchvision import transforms

from dataset.CUB200 import CUB_200
from dataset.ConText import ConText, MakeList, MakeListImage
from dataset.transform_func import make_transform

from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler
import numpy as np

def select_dataset(args):
    # MNIST
    if args.dataset == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor()
        ])
        dataset_train = MNIST(root="./data/mnist", train=True,  transform=transform, download=True)
        dataset_val   = MNIST(root="./data/mnist", train=False, transform=transform, download=True)
        sampler_train = None

    # CUB200
    elif args.dataset == "CUB200":
        dataset_train = CUB_200(args, train=True,  transform=make_transform(args, "train"))
        dataset_val   = CUB_200(args, train=False, transform=make_transform(args, "val"))
        sampler_train = None

    # ConText
    elif args.dataset == "ConText":
        train_list, val_list = MakeList(args).get_data()
        dataset_train = ConText(train_list, transform=make_transform(args, "train"))
        dataset_val   = ConText(val_list,   transform=make_transform(args, "val"))
        sampler_train = None

    # ImageNet
    elif args.dataset == "ImageNet":
        train_list, val_list = MakeListImage(args).get_data()
        dataset_train = ConText(train_list, transform=make_transform(args, "train"))
        dataset_val   = ConText(val_list,   transform=make_transform(args, "val"))
        sampler_train = None

    # blastocyst: 성공/실패 불균형 보정을 위해 WeightedRandomSampler 사용
    elif args.dataset == "blastocyst":
        transform_train = make_transform(args, "train")
        transform_val   = make_transform(args, "val")
        dataset_train = ImageFolder(root=f"{args.dataset_dir}/train", transform=transform_train)
        dataset_val   = ImageFolder(root=f"{args.dataset_dir}/val",   transform=transform_val)

        # 각 샘플의 레이블을 모아서 클래스별 개수 집계
        labels = [label for _, label in dataset_train]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[l] for l in labels]
        sampler_train = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    else:
        raise ValueError(f'unknown dataset "{args.dataset}"')

    return dataset_train, dataset_val, sampler_train