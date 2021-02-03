from torch.nn import Module
import torch.nn.functional as func
from torch import nn
import torch


class Swish(Module):
    """Swish激活函数"""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, input_):
        return input_ * torch.sigmoid(input_)


class Mish(Module):
    '''math:`Mish = x*tanh(ln(1+e^x))`'''

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, input_):
        return input_ * torch.tanh(func.softplus(input_))


class ChannelAttention(Module):
    """通道注意力"""

    def __init__(self, num_channels, r=4):
        """
        注意力
        :param num_channels: 通道数
        :param r: 下采样倍率
        """
        super(ChannelAttention, self).__init__()
        self.num_channels = num_channels
        self.layer = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // r, 1),
            Mish(),
            nn.Conv2d(self.num_channels // r, self.num_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, input_):
        return self.layer(torch.mean(input_, dim=[2, 3], keepdim=True)) * input_


class SpatialAttention(Module):
    """空间注意力"""

    def __init__(self, kernel_size=7):
        """
        注意力
        :param num_channels: 通道数
        :param r: 下采样倍率
        """
        super(SpatialAttention, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_):
        avg_out = torch.mean(input_, dim=1, keepdim=True)
        max_out, _ = torch.max(input_, dim=1, keepdim=True)
        return self.layer(torch.cat([avg_out, max_out], dim=1)) * input_


class ConvolutionLayer(Module):
    """卷积、批归一化、激活"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, action='mish'):
        """
        :param in_channels: 输入通道
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param padding:填充
        :param bias:偏置
        :param action:激活函数
        """
        super(ConvolutionLayer, self).__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.b = nn.BatchNorm2d(out_channels)
        self.is_action = True
        if action == 'mish':
            self.a = Mish()
        elif action == 'relu':
            self.a = nn.ReLU()
        elif action == 'leaky':
            self.a = nn.LeakyReLU(0.1)
        elif action == 'linear':
            self.is_action = False
        else:
            raise ValueError('Not this action')

    def forward(self, input_):
        if self.is_action:
            return self.a(self.b(self.c(input_)))
        else:
            return self.b(self.c(input_))


class Focus(Module):
    """下采样"""

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        """
        super(Focus, self).__init__()
        self.layer = ConvolutionLayer(in_channels * 4, out_channels,action='leaky')

    def forward(self, input_):
        return self.layer(torch.cat(
            (input_[:, :, ::2, ::2], input_[:, :, ::2, 1::2], input_[:, :, 1::2, ::2], input_[:, :, 1::2, 1::2]), 1
        ))


class Pool(Module):
    '''多尺度特征融合，借鉴Inception网络结构'''

    def __init__(self):
        super(Pool, self).__init__()
        self.max1 = nn.MaxPool2d(5, 1, 2)
        self.max2 = nn.MaxPool2d(9, 1, 4)
        self.max3 = nn.MaxPool2d(13, 1, 6)

    def forward(self, input_):
        return torch.cat((self.max1(input_), self.max2(input_), self.max3(input_), input_), dim=1)


class ResBlock(Module):
    """残差块,带注意力"""

    def __init__(self, channels, r=2):
        """
        :param channels: 通道数
        :param r: 通道缩减倍数，默认：2。
        """
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            ConvolutionLayer(channels, channels // r),
            ConvolutionLayer(channels // r, channels),
            ChannelAttention(channels),
            SpatialAttention(),
        )

    def forward(self, input_):
        return self.layer(input_) + input_


class Block(Module):

    def __init__(self, in_channels, out_channels, num_res):
        super(Block, self).__init__()
        res = []
        for i in range(num_res):
            res.append(ResBlock(in_channels))
        self.c1 = ConvolutionLayer(in_channels, out_channels, 3, 2)
        self.c2 = nn.Sequential(
            ConvolutionLayer(out_channels, in_channels, kernel_size=1, padding=0),
            *res,
            ConvolutionLayer(in_channels, in_channels, kernel_size=1, padding=0)
        )
        self.c3 = ConvolutionLayer(out_channels, in_channels, kernel_size=1, padding=0)
        self.c4 = ConvolutionLayer(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, input_):
        input_ = self.c1(input_)
        input_ = torch.cat((self.c2(input_), self.c3(input_)), dim=1)
        return self.c4(input_)


class FPN(Module):

    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        if in_channels != 2*out_channels:
            raise ValueError('The in_channels should be twice the out_channels')
        self.c1 = ConvolutionLayer(in_channels, out_channels, kernel_size=1, padding=0, action='leaky')
        self.c2 = ConvolutionLayer(in_channels, out_channels, kernel_size=1, padding=0, action='leaky')
        num_channels = int(out_channels+out_channels)
        self.c3 = nn.Sequential(
            ConvolutionLayer(num_channels, out_channels, kernel_size=1, padding=0, action='leaky'),
            ConvolutionLayer(out_channels, num_channels, action='leaky'),
            ConvolutionLayer(num_channels, out_channels, kernel_size=1, padding=0, action='leaky'),
            ConvolutionLayer(out_channels, num_channels, action='leaky'),
            ConvolutionLayer(num_channels, out_channels, kernel_size=1, padding=0, action='leaky'),
        )

    def forward(self, f1, f2):
        f1 = func.interpolate(self.c1(f1), scale_factor=2, mode='nearest')
        f2 = self.c2(f2)
        return self.c3(torch.cat([f1, f2], dim=1))
