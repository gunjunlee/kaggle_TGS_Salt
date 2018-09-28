import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import shy

nonlinearity = nn.ReLU

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = shy.layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bn=True, activation='relu')
        self.conv2 = shy.layer.Conv2d(channels, out_channels, kernel_size=3, padding=1, bn=True, activation='relu')
        
    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], dim=1)
        
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class Custom34(nn.Module):
    def __init__(self, num_classes=1):
        super(Custom34, self).__init__()

        # resnet = models.resnet34(pretrained=True)
        resnet = models.ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1)

        # self.first = nn.Sequential(
        #     shy.layer.Conv2d(3, 64, kernel_size=7, padding=3, bn=True, activation='relu'),
        # )

        self.first = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        self.center = nn.Sequential(
            shy.layer.Conv2d(512, 512, kernel_size=3, padding=1, bn=True, activation='relu'),
            shy.layer.Conv2d(512, 256, kernel_size=3, padding=1, bn=True, activation='relu'),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder5 = Decoder(256+512, 512, 64)
        self.decoder4 = Decoder(64+256, 256, 64)
        self.decoder3 = Decoder(64+128, 128, 64)
        self.decoder2 = Decoder(64+64, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        # Final Classifier
        self.logit = nn.Sequential(
            nn.Dropout2d(p=0.5),
            shy.layer.Conv2d(320, 64, kernel_size=3, padding=1, activation='relu'),
            shy.layer.Conv2d(64, num_classes, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder

        # x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.first(x)
        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        f = self.center(e5)
        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2)

        # Final Classification
        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ], dim=1)

        # f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        logit = self.logit(f)

        return logit

if __name__ == '__main__':
    net = Custom34()
    print(net(torch.ones((3, 3, 128, 128))).shape)
    import pdb
    pdb.set_trace()