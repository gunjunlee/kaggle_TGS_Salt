import os
import torch
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from math import ceil
import pdb

class Salt_dataset(torch.utils.data.Dataset):
    def __init__(self, dir_root, dir_image='image', dir_mask='mask', is_train=True, val_rate=0.2, transform=None):
        self.dir_root = dir_root
        self.dir_image = dir_image
        self.dir_mask = dir_mask
        self.is_train = is_train
        self.val_rate = val_rate
        self.transform = transform
        self.file_list = []
        
        for path, _, files in os.walk(os.path.join(dir_root, dir_image)):
            for file_ in files:
                self.file_list.append(file_)

        if self.dir_mask:
            cut = ceil(len(self) * val_rate)
            if self.is_train:
                self.file_list = self.file_list[:-cut]
            else:
                self.file_list = self.file_list[-cut:]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.dir_root, self.dir_image, self.file_list[idx])).convert("L")
        if self.dir_mask:
            mask = Image.open(os.path.join(self.dir_root, self.dir_mask, self.file_list[idx])).convert("L")
        else:
            mask = None

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)
        
        if len(sample) == 1:
            sample = sample, self.file_list[idx]

        return sample


def transform(sample):

    sample = totensor(sample)
    sample = normalize(sample, 0.5, 0.2)
    if sample['mask'] is not None:
        return sample['image'].contiguous(), sample['mask'].contiguous()
    else:
        return sample['image'].contiguous()

def totensor(sample):
    image = sample['image']
    mask = sample['mask']
    w, h = image.size
    image = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes())).float().div_(255)
    image = image.view(h, w, 1)
    image = image.permute((2, 0, 1))
    
    if mask  is not None:
        mask = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes())).long().div_(255)
        mask = mask.view(h, w)

    return {'image': image, 'mask': mask}

def normalize(sample, mean, std):
    image = sample['image']
    mask = sample['mask']
    image = (image - mean) / std
    return {'image': image, 'mask': mask}

def horizontal_flop(sample, prob=0.5):
    image = sample['image']
    mask = sample['mask']

    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if mask is not None:
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    
    return {'image': image, 'mask': mask}