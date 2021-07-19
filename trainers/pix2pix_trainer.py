"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel


class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)

#XXX        with open("/home/natsuki/hoge.log","a") as f:
#XXX            log = str(self.pix2pix_model.netG.model.conv1.weight).split("\n")[1]
#XXX            f.write(f"{__file__} {log}\n")

        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model
        if opt.isTrain:
            self.optimizer_G = self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, pred = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses

    def get_latest_losses(self):
        return {**self.g_losses }

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        new_lr = self.opt.lr/(2**epoch)
        if new_lr != self.old_lr:
            new_lr_G = new_lr
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
