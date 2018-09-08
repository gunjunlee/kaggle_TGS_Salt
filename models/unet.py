import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
import pdb

class Downsample_block(nn.Module):
    def __init__(self, in_planes, planes, out_planes):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3, 3), padding=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(planes, out_planes, kernel_size=(3, 3), padding=(1, 1), bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class Upsample_block(nn.Module):
    def __init__(self, in_planes, h_planes0, h_planes1, out_planes):
        super(Upsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, h_planes0, kernel_size=(3, 3), padding=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(h_planes0, h_planes1, kernel_size=(3, 3), padding=(1, 1), bias=True)
        self.trans_conv = nn.ConvTranspose2d(h_planes1, out_planes, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), output_padding=(0, 0))
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        if y is not None:
            diffx, diffy = y.shape[2]-x.shape[2], y.shape[3]-x.shape[3]
            x = F.pad(x, (diffx//2, ceil(diffx/2), diffy//2, ceil(diffy/2)))
            x = torch.cat([x, y], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.trans_conv(x)
        return x

class Last_conv_block(nn.Module):
    def __init__(self, in_planes, h_planes0, h_planes1, classes):
        super(Last_conv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, h_planes0, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(h_planes0, h_planes1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(h_planes1, classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        if y is not None:
            diffx, diffy = y.shape[2]-x.shape[2], y.shape[3]-x.shape[3]
            x = F.pad(x, (diffx//2, ceil(diffx/2), diffy//2, ceil(diffy/2)))
            x = torch.cat([x, y], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down1 = Downsample_block(1, 64, 64)
        self.down2 = Downsample_block(64, 128, 128)
        self.down3 = Downsample_block(128, 256, 256)
        self.down4 = Downsample_block(256, 512, 512)
        self.up1 = Upsample_block(512, 1024, 1024, 512)
        self.up2 = Upsample_block(1024, 512, 512, 256)
        self.up3 = Upsample_block(512, 256, 256, 128)
        self.up4 = Upsample_block(256, 128, 128, 64)
        self.last = Last_conv_block(128, 64, 64, 2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                print("initialize ", m)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                print("initialize ", m)

    def forward(self, x):
        y0 = self.down1(x)
        x = self.pool(y0)
        y1 = self.down2(x)
        x = self.pool(y1)
        y2 = self.down3(x)
        x = self.pool(y2)
        y3 = self.down4(x)
        x = self.pool(y3)
        x = self.up1(x)
        x = self.up2(x, y3)
        x = self.up3(x, y2)
        x = self.up4(x, y1)
        x = self.last(x, y0)
        return x

if __name__ == '__main__':
    # test
    net = Unet().cuda()
    pdb.set_trace()
    net(torch.ones([1, 1, 101, 101]).cuda())
    pass