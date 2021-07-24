"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import configargparse as argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from torchvision.models import resnet34
import timm
from collections import defaultdict, OrderedDict

class ResNet34Generator(BaseNetwork):
    def __init__(self, opt: argparse.Namespace) -> None:
        super().__init__()
        self.opt = opt
        self.model = resnet34(pretrained=False)
        if not opt.transfer_weights == "":
            weights = torch.load(opt.transfer_weights)
            # 一時的に最終層のshapeをload元に合わせる
            self.model.fc = nn.Linear(in_features=512, out_features=weights["model.fc.weight"].shape[0], bias=True)
            self.load_state_dict(weights)
            if self.opt.transfer_freeze:
                for name, param in self.model.named_parameters(): 
                    param.requires_grad = any( subname in name for subname in self.opt.transfer_unfreeze.split(",") )
        self.model.fc = nn.Linear(in_features=512, out_features=opt.out_features, bias=True)

    def forward(self, input: torch.Tensor, z=None) -> torch.Tensor:
        out = self.model(input)
        return out

class EfficientNetB0Generator(BaseNetwork):
    def __init__(self, opt: argparse.Namespace) -> None:
        super().__init__()
        self.opt = opt
        self.model = timm.create_model('efficientnet_b0', pretrained=False)
        if not opt.transfer_weights == "":
            weights = torch.load(opt.transfer_weights)
            # 一時的に最終層のshapeをload元に合わせる
            self.model.classifier = self.eff_fc_layer(weights["model.fc.weight"])
            self.load_state_dict(weights)
            if self.opt.transfer_freeze:
                for param in self.model.parameters(): # 最終層以外の重みを固定
                    param.requires_grad = False

        self.model.classifier = self.eff_fc_layer(opt.out_features)

    def eff_fc_layer(self, out_features: int):
        fc_list = [nn.Linear(in_features=1280, out_features=32, bias=True),
                   nn.Dropout(0.2),
                   nn.Linear(in_features=32, out_features=out_features, bias=True)]
        return nn.Sequential(*fc_list)

    def forward(self, input: torch.Tensor, z=None) -> torch.Tensor:
        out = self.model(input)
        return out