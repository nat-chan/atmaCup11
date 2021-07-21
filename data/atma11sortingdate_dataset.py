#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from data.atma11simple_dataset import Atma11SimpleDataset
import torch

class Atma11SortingDateDataset(Atma11SimpleDataset):
    def load_target(self, i: int) -> torch.Tensor:
        target_tensor = torch.Tensor([self.df.iloc[i]["sorting_date"]])
        return target_tensor