"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
import util.util as util
from torch import nn


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.netG = self.initialize_networks(opt)
        if opt.isTrain:
            self.criterion = nn.MSELoss()

    def forward(self, data, mode):
        if mode == 'generator': # train
            g_loss, pred = self.compute_generator_loss(data)
            return g_loss, pred
        elif mode == 'inference': # test
            with torch.no_grad():
                pred = self.netG(data["image"])
            return pred
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        # TODO 余裕があればAdamWとかSAMとか試す。（ムリかも）
        optimizer_G = torch.optim.Adam(G_params, lr=opt.lr, betas=(opt.beta1, opt.beta2))
        return optimizer_G

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        if not opt.isTrain or opt.continue_train:
            try:
                netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            except:
                print(f'Could not load {opt.which_epoch} model.') 
        return netG

    def compute_generator_loss(self, data):
        # TODO 将来的に、複数のfeatureを同時予測してラムダでパラメタつけしてもいい
        G_losses = dict()
        pred = self.netG(data["image"])
        target = data["target"]
        G_losses["MSE"] = self.criterion(pred, target)
        return G_losses, pred

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
