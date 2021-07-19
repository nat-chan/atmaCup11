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

class ResNet34Generator(BaseNetwork):
    def __init__(self, opt: argparse.Namespace) -> None:
        super().__init__()
        self.opt = opt
        self.model = resnet34(pretrained=False)
        if not opt.transfer_weights == "":
            weights = torch.load(opt.transfer_weights)
            self.model.load_state_dict(weights)
            for param in self.model.parameters(): # 最終層以外の重みを固定
                param.requires_grad = False
        self.model.fc = nn.Linear(in_features=512, out_features=opt.out_features, bias=True)

    def forward(self, input: torch.Tensor, z=None) -> torch.Tensor:
        out = self.model(input)
        return out