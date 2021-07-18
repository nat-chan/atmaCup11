#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import random

def make_deterministic(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True