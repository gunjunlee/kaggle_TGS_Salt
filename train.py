import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from dataloader import Data_loader, transform
from net import Unet

VAL_RATIO = 0.1
BATCH_SIZE = 32
NUM_PROCESSES = 8
MEAN, STD = (0.480,), (0.1337,)
EPOCHS = 5
LEARNING_RATE = 1e-3

if __name__ == '__main__':
    data_loader = {is_train: Data_loader('./data/train', 'images', 'masks',
         is_train=='train', VAL_RATIO, transform) for is_train in ['train', 'val']}

    pdb.set_trace()
    net = Unet().cuda()
    net = nn.DataParallel(net, [0, 1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    print("train start")
    for epoch in range(EPOCHS):
        train_running_loss = 0
        train_running_corrects = 0
        val_running_loss = 0
        val_running_corrects = 0

        # train
        net.train()

        gen_batch = data_loader.make_batch_from_file_list('./data/train/', 
                                                          'images',
                                                          'masks',
                                                          is_train=True,
                                                          batch_size=BATCH_SIZE,
                                                          num_processes=NUM_PROCESSES)
        total = ceil(data_loader.train_size/BATCH_SIZE)
        
        for batch_orig, batch_mask in tqdm(gen_batch, total=total):
            scheduler.step()
            optimizer.zero_grad()

            batch_orig = to_tensor_img(batch_orig).cuda()
            batch_orig = normalize(batch_orig, MEAN, STD)
            batch_mask = to_tensor_mask(batch_mask).cuda()

            with torch.set_grad_enabled(True):
                outputs = net(batch_orig)
                # pdb.set_trace()
                _, preds = torch.max(outputs, dim=1)
                try:
                    loss = criterion(outputs, batch_mask)
                except:
                    pdb.set_trace()
                    pass
                loss.backward()
                optimizer.step()
            train_running_corrects += torch.sum(preds == batch_mask).item()
            train_running_loss += loss.item() * batch_orig.size(0)

        # val
        net.eval()
        
        gen_batch = data_loader.make_batch_from_file_list('./data/train/', 
                                                          'images',
                                                          'masks',
                                                          is_train=False,
                                                          batch_size=BATCH_SIZE,
                                                          num_processes=NUM_PROCESSES)
        total = ceil(data_loader.val_size/BATCH_SIZE)
        for batch_orig, batch_mask in tqdm(gen_batch, total=total):
            batch_orig = to_tensor_img(batch_orig).cuda()
            batch_orig = normalize(batch_orig, MEAN, STD)
            batch_mask = to_tensor_mask(batch_mask).cuda()
            outputs = net(batch_orig)
            _, preds = torch.max(outputs, dim=1)
            val_running_corrects += torch.sum(preds == batch_mask).item()
            val_running_loss += loss.item() * batch_orig.size(0)

        print('train loss: {} \t acc: {}'.format(train_running_loss/train_size, train_running_corrects/(train_size*224*224)))
        print('test loss: {} \t acc: {}'.format(val_running_loss/val_size, val_running_corrects/(val_size*224*224)))

    
    print("train end")
        
