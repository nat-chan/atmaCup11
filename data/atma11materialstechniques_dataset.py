#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import *
import configargparse as argparse
from data.atma11simple_dataset import Atma11SimpleDataset
import torch.nn.functional as F
import torch
import pandas as pd

class Atma11MaterialsTechniquesDataset(Atma11SimpleDataset):
    def initialize(self, opt: argparse.Namespace) -> None:
        super().initialize(opt)
        self.outer_df = pd.read_pickle( self.root / f"materialstechniques_100.pkl" )

    def load_target(self, i: int) -> torch.Tensor:
        target_tensor = torch.tensor(self.outer_df.loc[self.df.iloc[i]["object_id"]].values)
        return target_tensor


