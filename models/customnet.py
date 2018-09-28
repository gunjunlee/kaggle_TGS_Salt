import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import shy

nonlinearity = nn.ReLU

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

        resnet = models.resnet34(pretrained=True)

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
            shy.layer.Conv2d(64, 64, kernel_size=3, padding=1, activation='relu'),
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
        f = d1
        f = F.dropout2d(f, p=0.5)
        logit = self.logit(f)

        return logit

if __name__ == '__main__':
    net = Custom34()
    print(net(torch.ones((3, 3, 256, 256))).shape)
    import pdb
    pdb.set_trace()