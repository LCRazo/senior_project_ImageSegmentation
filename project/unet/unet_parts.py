""" Parts of the U-Net model """
#imports neural network module
import torch
import torch.nn as nn
#functions that operates on tensors
import torch.nn.functional as F

#represents a double convolution block used in U-net
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    #initializes double convolution block with input, output and intermediate channels
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        #Sets intermediate channels to be equal to the number of output channels if it isnt sepcified
        if not mid_channels:
            mid_channels = out_channels

        #Defines a sequential module consisting of 2 convolutional layers followed by normalization and reLu activation
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    #Defines the foward pass and specifies how input data flows through double convolution block
    def forward(self, x):
        #passes the input tensor through sequential module and returns output tensor after the double convolution operation
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    #initializes downsampling block
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #defines sequential module consisting of 2 operations
        self.maxpool_conv = nn.Sequential(
            #downsamples input tensor by factor of 2
            nn.MaxPool2d(2),
            #double convolution block performs two consecutive convolution operations followed by activation and normalization
            DoubleConv(in_channels, out_channels)
        )

    #defines foward pass specifies how input data x flows through the downsampling block
    def forward(self, x):


        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    # initializes upsampling block
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


#output convolution layer in neural network model
class OutConv(nn.Module):
    #initializes output convolutional layer
    def __init__(self, in_channels, out_channels):
        #calls constructor of the parent class nn.Module to initialize
        super(OutConv, self).__init__()
        #Creates a convolutional layer with 2d, performs 1x1 concolutions
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, x):
        return self.conv(x)
