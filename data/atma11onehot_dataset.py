#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from data.atma11simple_dataset import Atma11SimpleDataset
import torch.nn.functional as F
import torch

class Atma11OneHotDataset(Atma11SimpleDataset):
    def load_target(self, i: int) -> torch.Tensor:
        target_tensor = F.one_hot(torch.tensor(self.df.iloc[i]["target"]), num_classes=4).float()
        return target_tensor


