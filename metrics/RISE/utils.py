""" 
Code was partially taken and adapted from the following paper:

Petsiuk, V., Das, A., & Saenko, K. (2018). 
Rise: Randomized input sampling for explanation of black-box models. 
arXiv preprint arXiv:1806.07421.

Code available at: https://github.com/eclique/RISE
Commit: d91ea00 on Sep 17, 2018
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from PIL import Image


# Dummy class to store arguments
class Dummy:
    pass


# Function that opens image from disk, normalizes it and converts to tensor
read_tensor = transforms.Compose(
    [
        lambda x: Image.open(x),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        lambda x: torch.unsqueeze(x, 0),
    ]
)


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = torch.Tensor.cpu(inp).numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


# Given label number returns class name
def get_class_name(c):
    labels = np.loadtxt("synset_words.txt", str, delimiter="\t")
    return " ".join(labels[c].split(",")[0].split()[1:])


# Image preprocessing function
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalization for ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)