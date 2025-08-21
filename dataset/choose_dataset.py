from dataset.mnist import MNIST
from dataset.CUB200 import CUB_200
from dataset.ConText import ConText, MakeList, MakeListImage
from dataset.transform_func import make_transform


def select_dataset(args):
    # Blastocyst만 지원하도록 (train, val, test로 제대로 학습할 수 있도록)
    if args.dataset == "MNIST":
        dataset_train = MNIST('./data/mnist', train=True, download=True, transform=make_transform(args, "train"))
        dataset_val = MNIST('./data/mnist', train=False, transform=make_transform(args, "val"))
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
    if args.dataset == "Blastocyst":
        args.category = ["success", "failure"]          
        args.num_classes = len(args.category)          

        tf_train = make_transform(args, "train")  # train은 학습 전용 변환 사용(확률적 증강) 
        tf_eval  = make_transform(args, "val")   # val, test는 고정 변환 사용

        train, val, test = MakeListImage(args).get_data()
        if len(test) == 0:
            raise RuntimeError("test 폴더가 비어있거나 없습니다. 독립 테스트셋을 준비하세요.")
        dataset_train = ConText(train, transform=tf_train)
        dataset_val   = ConText(val,   transform=tf_eval)
        dataset_test  = ConText(test,  transform=tf_eval)
        return dataset_train, dataset_val, dataset_test