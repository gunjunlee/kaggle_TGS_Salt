import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms

class Unet(nn.Module):
    def __init__(self, feature_scale=1, n_classes=1, is_deconv=True, in_channels=3,
                is_bn=False, filters=None):
        super(Unet, self).__init__()
        
        if filters is None:
            filters = [64, 128, 256, 512, 1024]
        
        filters = [x // feature_scale for x in filters]

        self.down1 = UnetDown(in_channels, filters[0], is_bn)
        self.down2 = UnetDown(filters[0], filters[1], is_bn)
        self.down3 = UnetDown(filters[1], filters[2], is_bn)
        self.down4 = UnetDown(filters[2], filters[3], is_bn)

        self.center = UnetConvBlock(filters[3], filters[4], is_bn)

        self.up1 = UnetUp(filters[4], filters[3], is_deconv=is_deconv, is_bn=is_bn)
        self.up2 = UnetUp(filters[3], filters[2], is_deconv=is_deconv, is_bn=is_bn)
        self.up3 = UnetUp(filters[2], filters[1], is_deconv=is_deconv, is_bn=is_bn)
        self.up4 = UnetUp(filters[1], filters[0], is_deconv=is_deconv, is_bn=is_bn)
        
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)
    
    def forward(self, x):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)
        x = self.center(x)
        x = self.up1(skip4, x)
        x = self.up2(skip3, x)
        x = self.up3(skip2, x)
        x = self.up4(skip1, x)
        x = self.final(x)
        return x


class UnetDown(nn.Module):
    def __init__(self, in_size, out_size, is_bn):
        super(UnetDown, self).__init__()
        
        self.conv = UnetConvBlock(in_size, out_size, is_bn, num_layers=2)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        skip = self.conv(x)
        output = self.pool(skip)
        return skip, output

class UnetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=False, residual_size=None, is_bn=False):
        super(UnetUp, self).__init__()
        if residual_size is None:
            residual_size = out_size
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)
            self.conv = UnetConvBlock(in_size + residual_size, out_size, is_bn=is_bn, num_layers=2)
        else:
            self.up = nn.UpSample(scale_factor=2, mode='bilinear')
            self.conv = UnetConvBlock(in_size + residual_size, out_size, is_bn=is_bn, num_layers=2)
    
    def forward(self, skip, x):
        upsample = self.up(x)
        output = self.conv(torch.cat([skip, upsample], dim=1))
        return output

class UnetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, is_bn, num_layers=2):
        super(UnetConvBlock, self).__init__()
        self.convs = nn.ModuleList()
        if is_bn:
            conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU()
            )
            self.convs.append(conv)
            for i in range(1, num_layers):
                conv = nn.Sequential(
                    nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU()
                )
                self.convs.append(conv)
        else:
            conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )
            self.convs.append(conv)
            for i in range(1, num_layers):
                conv = nn.Sequential(
                    nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()
                )
                self.convs.append(conv)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

if __name__ == '__main__':
    net = Unet()
    net(torch.ones((1, 3, 224, 224)))
