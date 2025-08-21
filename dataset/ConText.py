from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tools.prepare_things import get_name
import os
import torch
import numpy as np


class MakeList(object):
    """
    this class used to make list of data for model train and test, return the root name of each image
    root: txt file records condition for every cxr image
    """
    def __init__(self, args, ratio=0.8):
        self.image_root = args.dataset_dir
        self.all_image = get_name(self.image_root, mode_folder=False)
        self.category = sorted(set([i[:i.find('_')] for i in self.all_image]))

        for c_id, c in enumerate(self.category):
            print(c_id, '\t', c)

        self.ration = ratio

    def get_data(self):
        all_data = []
        for img in self.all_image:
            label = self.deal_label(img)
            all_data.append([os.path.join(self.image_root, img), label])
        train, val = train_test_split(all_data, random_state=1, train_size=self.ration)
        return train, val

    def deal_label(self, img_name):
        categoty_no = img_name[:img_name.find('_')]
        back = self.category.index(categoty_no)
        return back


class MakeListImage():
    """
    this class used to make list of data for ImageNet
    """
    # Blastocyst 데이터셋 test 폴더 사용하도록 수정
    def __init__(self, args):
        self.image_root = args.dataset_dir
        train_dir = os.path.join(self.image_root, "train")
        self.category = get_name(train_dir) or []
        self.used_cat = self.category[:args.num_classes]

    def get_data(self):
        train = self.get_img(self.used_cat, "train")
        val   = self.get_img(self.used_cat, "val")
        test  = self.get_img(self.used_cat, "test")
        return train, val, test

    def get_img(self, folders, phase):
        record = []
        root_phase = os.path.join(self.image_root, phase)
        if not os.path.isdir(root_phase):
            return record
        for cls in folders:
            cls_dir = os.path.join(root_phase, cls)
            for img in get_name(cls_dir, mode_folder=False):
                record.append([os.path.join(cls_dir, img), self.deal_label(cls)])
        return record

    def deal_label(self, cls_name):
        return self.used_cat.index(cls_name)


class ConText(Dataset):
    """read all image name and label"""
    def __init__(self, data, transform=None):
        self.all_item = data
        self.transform = transform

    def __len__(self):
        return len(self.all_item)

    def __getitem__(self, item_id):  # generate data when giving index
        while not os.path.exists(self.all_item[item_id][0]):
            raise ("not exist image:" + self.all_item[item_id][0])
        image_path = self.all_item[item_id][0]
        image = Image.open(image_path).convert('RGB')
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.all_item[item_id][1]
        label = torch.tensor(label, dtype=torch.long)
        return {"image": image, "label": label, "names": image_path}