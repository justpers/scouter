from torchvision.datasets import MNIST
from torchvision import transforms

from dataset.CUB200 import CUB_200
from dataset.ConText import ConText, MakeList, MakeListImage
from dataset.transform_func import make_transform

def select_dataset(args):
    if args.dataset == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor()
        ])
        dataset_train = MNIST(root="./data/mnist", train=True, transform=transform, download=True)
        dataset_val   = MNIST(root="./data/mnist", train=False, transform=transform, download=True)
        return dataset_train, dataset_val
    if args.dataset == "CUB200":
        dataset_train = CUB_200(args, train=True, transform=make_transform(args, "train"))
        dataset_val = CUB_200(args, train=False, transform=make_transform(args, "val"))
        return dataset_train, dataset_val
    if args.dataset == "ConText":
        train, val = MakeList(args).get_data()
        dataset_train = ConText(train, transform=make_transform(args, "train"))
        dataset_val = ConText(val, transform=make_transform(args, "val"))
        return dataset_train, dataset_val
    if args.dataset == "ImageNet":
        train, val = MakeListImage(args).get_data()
        dataset_train = ConText(train, transform=make_transform(args, "train"))
        dataset_val = ConText(val, transform=make_transform(args, "val"))
        return dataset_train, dataset_val

    raise ValueError(f'unknown {args.dataset}')