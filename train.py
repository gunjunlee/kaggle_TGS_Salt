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
from models import Unet, LinkNet34, Custom34, ResUnet
from metric import dice_loss, iou
from termcolor import colored
from lovasz_losses import lovasz_hinge

VAL_RATIO = 0.1
BATCH_SIZE = 32
NUM_PROCESSES = 8
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EPOCHS = 120
LEARNING_RATE = 1e-2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs        = torch.sigmoid(logits)
        probs_flat   = probs.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)

if __name__ == '__main__':
    dataset = {phase: Salt_dataset('./data/train', 'images', 'masks',
         phase=='train', VAL_RATIO, transform) for phase in ['train', 'val']}
    dataloader = {phase: torch.utils.data.DataLoader(dataset[phase],
         batch_size=BATCH_SIZE, shuffle=True, num_workers=32)
         for phase in ['train', 'val']}

    # net = Unet(n_classes=1, in_channels=1, is_bn=True)
    # net = LinkNet34(num_channels=3, num_classes=1, pretrained=True)
    # net = Custom34()
    net = ResUnet()

    # freezing
    # for parameters in [net.firstconv, net.firstbn, net.firstmaxpool]:
    #     for param in parameters.parameters():
    #         param.requires_grad = False

    print('data parallel')
    net = nn.DataParallel(net.cuda())
    print('data parallel end')
    criterion = BCELoss2d()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)
    
    # net.load_state_dict(torch.load('ckpt/model.pth'))

    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-6)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    
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
            # pdb.set_trace()

            batch_image = batch_image.cuda()
            batch_mask = batch_mask.cuda()
            with torch.set_grad_enabled(True):
                outputs = net(batch_image).squeeze(dim=1)

                if epoch < 120:
                    loss = criterion(outputs, batch_mask)
                else:
                    loss = lovasz_hinge(outputs, batch_mask)

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

        # scheduler.step()
        scheduler.step(train_running_loss/len(dataset['train']))
        
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
            torch.save(net.state_dict(), 'ckpt/model2.pth')
            print(colored('model saved', 'red'))

    print("train end")
        
