import os
import torch
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from PIL import Image
from math import ceil
import pdb

class Salt_dataset(torch.utils.data.Dataset):
    def __init__(self, dir_root, dir_image='images', dir_mask='masks', is_train=True, val_rate=0.2, transform=None):
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
        # pdb.set_trace()
        image = Image.open(os.path.join(self.dir_root, self.dir_image, self.file_list[idx])).convert('L')
        image = image.resize((128, 128), resample=Image.NEAREST)
        mask = None
        if self.dir_mask:
            mask = Image.open(os.path.join(self.dir_root, self.dir_mask, self.file_list[idx])).convert('L')
            mask = mask.resize((128, 128), resample=Image.NEAREST)
            mask = np.array(mask)/255
            mask = torch.tensor(mask).long()
        else:
            mask = self.file_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, mask

if __name__ == '__main__':
    MEAN, STD = [0.5], [1]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    dataset = Salt_dataset('data/train', transform=transform)
    dataset[1]