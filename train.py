import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from math import ceil
import pdb
import os
import time
import random
from utils import *
from dataloader import Salt_dataset
from models.unet import Unet
from models.linknet import LinkNet34
from metric import dice_loss, iou
from termcolor import colored

VAL_RATIO = 0.1
BATCH_SIZE = 32
NUM_PROCESSES = 8
MEAN, STD = [0.5], [1]
EPOCHS = 120
LEARNING_RATE = 1e-3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

if __name__ == '__main__':
    dataset = {phase: Salt_dataset('./data/train', 'images', 'masks',
         phase=='train', VAL_RATIO, transform) for phase in ['train', 'val']}
    dataloader = {phase: torch.utils.data.DataLoader(dataset[phase],
         batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
         for phase in ['train', 'val']}

    # net = Unet(n_classes=1, in_channels=1, is_bn=True)
    net = LinkNet34(num_channels=1, num_classes=1, pretrained=True)
    net = nn.DataParallel(net.cuda())
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    min_iou = 100

    print("train start")
    for epoch in range(EPOCHS):
        print('epoch: {}'.format(epoch))
        train_running_loss = 0
        train_running_dice_loss = 0
        train_running_corrects = 0
        train_iou = np.zeros(20)

        val_running_loss = 0
        val_running_dice_loss = 0
        val_running_corrects = 0
        val_iou = np.zeros(20)

        # train
        net.train()
        for batch_image, batch_mask in tqdm(dataloader['train']):
            optimizer.zero_grad()

            batch_image = batch_image.cuda()
            batch_mask = batch_mask.cuda()
            # pdb.set_trace()
            with torch.set_grad_enabled(True):
                outputs = net(batch_image).squeeze(dim=1)

                loss = criterion(outputs, batch_mask.float())\
                    + dice_loss(outputs, batch_mask)

                if loss.item() > 2:
                    print(colored(loss.item(), 'red'))
                loss.backward()
                optimizer.step()
            train_running_corrects += torch.sum((outputs>0.5) == (batch_mask>0.5)).item()
            train_running_loss += loss.item() * batch_image.size(0)
            train_running_dice_loss += dice_loss(outputs, batch_mask).item() * batch_image.size(0)
            train_iou += iou(outputs, batch_mask) * batch_image.size(0)

        # val
        net.eval()
        for batch_image, batch_mask in tqdm(dataloader['val']):
            batch_image = batch_image.cuda()
            batch_mask = batch_mask.cuda()
            outputs = net(batch_image).squeeze(dim=1)

            loss = criterion(outputs, batch_mask.float())\
                    + dice_loss(outputs, batch_mask)

            val_running_corrects += torch.sum((outputs>0.5) == (batch_mask>0.5)).item()
            val_running_loss += loss.item() * batch_image.size(0)
            val_running_dice_loss += dice_loss(outputs, batch_mask).item() * batch_image.size(0)
            val_iou += iou(outputs, batch_mask) * batch_image.size(0)

        scheduler.step()
        # scheduler.step(train_running_loss/len(dataset['train']))
        
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        print('train loss: {} \t dice loss: {} \t\n iou: {} \t acc: {}'.format(
            train_running_loss/len(dataset['train']),
            train_running_dice_loss/len(dataset['train']),
            train_iou/len(dataset['train']),
            train_running_corrects/(len(dataset['train'])*128*128)))

        print('val loss: {} \t dice loss: {} \t\n iou: {} \t acc: {}'.format(
            val_running_loss/len(dataset['val']),
            val_running_dice_loss/len(dataset['val']),
            val_iou/len(dataset['val']),
            val_running_corrects/(len(dataset['val'])*128*128)))

        if val_iou.max() > min_iou:
            min_iou = val_iou.max()
            torch.save(net.state_dict(), 'ckpt/unet-bn-inch1-plateau.pth')
            print(colored('model saved', 'red'))

    print("train end")
        
