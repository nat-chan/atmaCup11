"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import models.networks.standard as AlacGAN
import json


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar

class I2VEncoder(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        tag_path = "resources/tag_list.json"
        param_path = "resources/illust2vec_tag_ver200.pth"
        if tag_path is not None:
            tags = json.loads(open(tag_path, 'r').read())
            assert(len(tags) == 1539)
            self.tags = np.array(tags)
            self.index = {t: i for i, t in enumerate(tags)}
        inplace = True
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv6_1 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu6_1 = nn.ReLU(inplace)
        self.conv6_2 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu6_2 = nn.ReLU(inplace)
        self.conv6_3 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu6_3 = nn.ReLU(inplace)
        self.conv6_4 = nn.Conv2d(1024, 1539, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool6 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.prob = nn.Sigmoid()
        self.load_state_dict(torch.load(param_path))
        self.register_buffer('mean', torch.FloatTensor([164.76139251, 167.47864617, 181.13838569]).view(1, 3, 1, 1))
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            if x.size(2) != 224 or x.size(3) != 224:
                x = F.interpolate(x, size=(224, 224), mode='bilinear')
            x = x.mul(0.5).add(0.5).mul(255) # (-1,1)->(0,255)
            x = x - self.mean 
            x = self.conv1_1(x)
            x = self.relu1_1(x)
            x = self.pool1(x)
            x = self.conv2_1(x)
            x = self.relu2_1(x)
            x = self.pool2(x)
            x = self.conv3_1(x)
            x = self.relu3_1(x)
            x = self.conv3_2(x)
            x = self.relu3_2(x)
            x = self.pool3(x)
            x = self.conv4_1(x)
            x = self.relu4_1(x)
            x = self.conv4_2(x)
            x = self.relu4_2(x)
            x = self.pool4(x)
            x = self.conv5_1(x)
            x = self.relu5_1(x)
            x = self.conv5_2(x)
            x = self.relu5_2(x)
            x = self.pool5(x)
            x = self.conv6_1(x)
            x = self.relu6_1(x)
            x = self.conv6_2(x)
            x = self.relu6_2(x)
            x = self.conv6_3(x)
            x = self.relu6_3(x)
            x = self.conv6_4(x)
            x = self.pool6(x)
            return x