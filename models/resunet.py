import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import shy

nonlinearity = nn.ReLU

class BatchActivate(nn.Module):
    def __init__(self, channels):
        super(BatchActivate, self).__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = None
        if activation == True:
            self.act = BatchActivate(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, batch_activate=False):
        super(ResidualBlock, self).__init__()
        self.bnact = BatchActivate(channels)
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, padding=1, activation=False)
        self.act = None
        if batch_activate:
            self.act = BatchActivate(channels)
    
    def forward(self, x):
        y = self.bnact(x)
        y = self.conv1(y)
        y = self.conv2(y)
        x = x + y
        if self.act is not None:
            x = self.act(x)
        return x


class ResUnet(nn.Module):
    def __init__(self, start_neurons=16, num_classes=1):
        super(ResUnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, start_neurons, kernel_size=3, padding=1),
            ResidualBlock(start_neurons),
            ResidualBlock(start_neurons, True)
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(start_neurons, start_neurons*2, kernel_size=3, padding=1),
            ResidualBlock(start_neurons*2),
            ResidualBlock(start_neurons*2, True),
        )
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(start_neurons*2, start_neurons*4, kernel_size=3, padding=1),
            ResidualBlock(start_neurons*4),
            ResidualBlock(start_neurons*4, True),
        )
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.5)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(start_neurons*4, start_neurons*8, kernel_size=3, padding=1),
            ResidualBlock(start_neurons*8),
            ResidualBlock(start_neurons*8, True),
        )
        self.pool4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.5)
        )

        self.convm = nn.Sequential(
            nn.Conv2d(start_neurons*8, start_neurons*16, kernel_size=3, padding=1),
            ResidualBlock(start_neurons*16),
            ResidualBlock(start_neurons*16, True),
        )

        self.deconv4 = nn.ConvTranspose2d(start_neurons*16, start_neurons*8, kernel_size=2, stride=2)
        self.uconv4 = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(start_neurons*16, start_neurons*8, kernel_size=3, padding=1),
            ResidualBlock(start_neurons*8),
            ResidualBlock(start_neurons*8, True)
        )

        self.deconv3 = nn.ConvTranspose2d(start_neurons*8, start_neurons*4, kernel_size=2, stride=2)
        self.uconv3 = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(start_neurons*8, start_neurons*4, kernel_size=3, padding=1),
            ResidualBlock(start_neurons*4),
            ResidualBlock(start_neurons*4, True)
        )

        self.deconv2 = nn.ConvTranspose2d(start_neurons*4, start_neurons*2, kernel_size=2, stride=2)
        self.uconv2 = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(start_neurons*4, start_neurons*2, kernel_size=3, padding=1),
            ResidualBlock(start_neurons*2),
            ResidualBlock(start_neurons*2, True)
        )

        self.deconv1 = nn.ConvTranspose2d(start_neurons*2, start_neurons, kernel_size=2, stride=2)
        self.uconv1 = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(start_neurons*2, start_neurons, kernel_size=3, padding=1),
            ResidualBlock(start_neurons),
            ResidualBlock(start_neurons, True)
        )

        self.output = nn.Conv2d(start_neurons, num_classes, kernel_size=1)

    def forward(self, x):
        import pdb
        pdb.set_trace()
        # encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        # center
        convm = self.convm(pool4)

        # decoder
        deconv4 = self.deconv4(convm)
        uconv4 = self.uconv4(torch.cat([deconv4, conv4], dim=1))

        deconv3 = self.deconv3(uconv4)
        uconv3 = self.uconv3(torch.cat([deconv3, conv3], dim=1))

        deconv2 = self.deconv2(uconv3)
        uconv2 = self.uconv2(torch.cat([deconv2, conv2], dim=1))

        deconv1 = self.deconv1(uconv2)
        uconv1 = self.uconv1(torch.cat([deconv1, conv1], dim=1))
        
        output = self.output(uconv1)

        return output

if __name__ == '__main__':
    net = ResUnet()
    print(net(torch.ones((3, 3, 128, 128))).shape)