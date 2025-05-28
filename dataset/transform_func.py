import torch
from tools.image_aug import ImageAugment
import torchvision.transforms.functional as F
from collections.abc import Sequence, Iterable
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class Resize(object):
    """class for resize images. """
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        # torchvision.functional.resize는 PIL Image나 Tensor를 받지만,
        # 여기서는 PIL Image → numpy array 로 바꿔줍니다.
        return np.array(F.resize(image, self.size, self.interpolation))

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + f'(size={self.size}, interpolation={interpolate_str})'


class Aug(object):
    def __init__(self, aug: bool, level: str = "base"):  # level = base|strong
        self.aug = aug
        self.level = level

    def __call__(self, image):
        if not self.aug:
            return image

        if self.level == "strong":
            seq = iaa.Sequential([
                iaa.SomeOf((1, 3), [
                    iaa.Fliplr(0.5), iaa.Flipud(0.3),
                    iaa.Affine(rotate=(-25, 25), scale=(0.9, 1.1)),
                    iaa.LinearContrast((0.7, 1.4)),
                    iaa.AddToBrightness((-30, 30)),
                    iaa.AdditiveGaussianNoise(scale=(0, 0.03*255))
                ])
            ])
        else:  # base
            seq = iaa.Sequential([
                iaa.SomeOf((0, 2), [
                    iaa.Fliplr(0.5),
                    iaa.Affine(rotate=(-15, 15)),
                    iaa.LinearContrast((0.8, 1.2)),
                    iaa.AddToBrightness((-20, 20))
                ])
            ])

        # imgaug은 (H, W, C) numpy array 를 받아서 동일한 shape 로 반환합니다.
        arr = image if isinstance(image, np.ndarray) else np.array(image)
        aug_arr = seq(image=arr)
        return aug_arr

    def __repr__(self):
        return self.__class__.__name__ + 'Augmentation function'


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor."""
    def __call__(self, image):
        if image.ndim == 2:
            # grayscale → (H, W, 1)
            image = image[:, :, None]
        # (H, W, C) → (C, H, W), 0~255 → 0.0~1.0
        tensor = torch.from_numpy((image / 255.0).transpose(2, 0, 1).copy()).float()
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    ' + repr(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgs):
        # imgs는 tensor (C, H, W)
        return F.normalize(imgs, mean=self.mean, std=self.std)


def make_transform(args, mode):
    normalize_value = {
        "MNIST":    [[0.1307],         [0.3081]],
        "CUB200":   [[0.485,0.456,0.406],[0.229,0.224,0.225]],
        "ConText":  [[0.485,0.456,0.406],[0.229,0.224,0.225]],
        "ImageNet": [[0.485,0.456,0.406],[0.229,0.224,0.225]],
        "blastocyst": [[0.485,0.456,0.406],[0.229,0.224,0.225]],
    }
    mean, std = normalize_value[args.dataset]
    normalize = Compose([ ToTensor(), Normalize(mean, std) ])

    if mode == "train":
        return Compose([
            Resize((args.img_size, args.img_size)),
            Aug(args.aug, level=getattr(args, "aug_level", "base")),  
            normalize,
        ])
    elif mode == "val":
        return Compose([
            Resize((args.img_size, args.img_size)),
            normalize,
        ])
    else:
        raise ValueError(f"unknown mode {mode}")