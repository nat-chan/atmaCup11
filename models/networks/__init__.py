"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from models.networks.base_network import BaseNetwork
from models.networks.generator import *
import util.util as util


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()
    netG_cls = find_network_using_name(opt.netG, 'generator')
    parser = netG_cls.modify_commandline_options(parser, is_train)
    return parser


def create_network(cls, opt):
    net = cls(opt)

#XXX    with open("/home/natsuki/hoge.log","a") as f:
#XXX        log = str(net.model.conv1.weight).split("\n")[1]
#XXX        f.write(f"{__file__} {log}\n")
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    if opt.transfer_weights == "":
        net.init_weights(opt.init_type, opt.init_variance)
#XXX    with open("/home/natsuki/hoge.log","a") as f:
#XXX        log = str(net.model.conv1.weight).split("\n")[1]
#XXX        f.write(f"{__file__} {log}\n")
    return net


def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator')
    return create_network(netG_cls, opt)


def define_D(opt):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt)


def define_E(opt):
    # there exists only one encoder type
    netE_cls = find_network_using_name(opt.netE, 'encoder')
    return create_network(netE_cls, opt)