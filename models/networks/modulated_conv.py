import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, kernel_size=1, padding=0, bias=False, demodulate=True):
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
#        self.padding = kernel_size // 2
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channel))
        else:
            self.bias = None
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size})'

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, self.bias, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out