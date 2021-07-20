"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE, get_nonspade_norm_layer
from models.networks.modulated_conv import ModulatedConv2d
import math

class UpBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=opt.resize_mode, align_corners=opt.resize_align_corners)
        self.to_rgb = ToRGB(fout)
        self.spaderesblk = SPADEResnetBlock(fin, fout, opt)

    def forward(self, x, seg, append=None, style=None, skip=None):
        x = self.up(x)
        x = self.spaderesblk(x, seg, append=append, style=style)
        out = self.to_rgb(x)
        if skip is not None:
            skip = self.up(skip)
            out = out + skip
        return x, out

class DownBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_Gdown)
        if opt.config_Gdown == "stride":
            self.model = nn.Sequential(
                norm_layer(nn.Conv2d(fin, fout, kernel_size=3, padding=1, stride=2, bias=False)),
                nn.LeakyReLU(0.2, True),
            )
        elif opt.config_Gdown == "maxpool":
            self.model = nn.Sequential(
                norm_layer(nn.Conv2d(fin, fout, kernel_size=3, padding=1, bias=False)),
                nn.MaxPool2d(2),
                nn.LeakyReLU(0.2, True),
            )

    def forward(self, x):
        opt = self.opt
        return self.model(x)

class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.conv_img = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, input):
        out = F.leaky_relu(input, 2e-1)
        out = self.conv_img(out)
        out = F.tanh(out)
        return out

class ToRGB2(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.conv_img = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, input):
        out = F.leaky_relu(input, 2e-1)
        out = self.conv_img(out)
        return out

class SPADESimpleBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fout, kernel_size=3, padding=1)
        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc, mode=opt.resize_mode, align_corners=opt.resize_align_corners)
    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, append=None):
        out = self.conv_0(self.actvn(self.norm_0(x, seg, append=append)))
        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1, inplace=True)

class ModulatedSPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, style_dim, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = ModulatedConv2d(fin, fmiddle, style_dim=style_dim, kernel_size=3, padding=1, bias=False)
        self.conv_1 = ModulatedConv2d(fmiddle, fout, style_dim=style_dim, kernel_size=3, padding=1, bias=False)
        if self.learned_shortcut:
            self.conv_s = ModulatedConv2d(fin, fout, style_dim=style_dim, kernel_size=1, padding=0, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc, mode=opt.resize_mode, align_corners=opt.resize_align_corners)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc, mode=opt.resize_mode, align_corners=opt.resize_align_corners)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc, mode=opt.resize_mode, align_corners=opt.resize_align_corners)
    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, style, append=None):
        x_s = self.shortcut(x, seg, style, append=append)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg, append=append)), style)
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, append=append)), style)
        out = x_s + dx
        return out

    def shortcut(self, x, seg, style, append=None):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, append=append), style)
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc, mode=opt.resize_mode, align_corners=opt.resize_align_corners, style_dim=opt.z_dim)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc, mode=opt.resize_mode, align_corners=opt.resize_align_corners, style_dim=opt.z_dim)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc, mode=opt.resize_mode, align_corners=opt.resize_align_corners, style_dim=opt.z_dim)
    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, append=None, style=None):
        x_s = self.shortcut(x, seg, append=append, style=style)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg, append=append, style=style)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, append=append, style=style)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg, append=None, style=None):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, append=append, style=style))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
