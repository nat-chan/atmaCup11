#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import numpy as np
path = "/data/natsuki/dataset_atmaCup11/checkpoints/atma11simple_j4nofreeze_all/epoch10_all_test_features.csv"
df = pd.read_csv(path)
sub = df.drop(columns="object_id").rename(columns=lambda x: "target").clip(lower=0, upper=3)
# %%
sub.min()
sub.max()
# %%
sub.to_csv("j4nofreeze.csv", index=False)
# %%