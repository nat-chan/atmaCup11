#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import os
from os import path
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from pathlib import Path

# %%

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

# test
predicts = defaultdict(list) #type: ignore
targets = defaultdict(list) #type: ignore
with tqdm(dataloader, dynamic_ncols=True) as pbar:
    for i, data_i in enumerate(pbar):
        # 複数のfeatureのときは　.cpu().detach().numpy()
        # np.mean(list_of_numpy, axis=0) で featureの次元を保持したままmeanが取れる
        object_id = data_i["path"][0].split("/")[-1][:-4]
        predict = model(data_i, mode='inference').item()
        predicts[object_id].append(predict)
        if opt.isVal:
            target = data_i["target"].item()
            targets[object_id].append(target)

losses = list() #type: ignore
with open(Path(opt.checkpoints_dir)/opt.name/f"epoch{opt.which_epoch}_{opt.df_csv[:-4]}_features.csv", "w") as f:
    f.write(f"object_id, " + ", ".join(f"{opt.name}_e{opt.which_epoch}_{i}" for i in range(opt.out_features)) + "\n") # TODO 複数次元のfeature
    for object_id in predicts:
        tta_predict = np.mean(predicts[object_id], axis=0) # のちの複数次元を見越して、1次元の時はfloat帰ってくる
        f.write(f"{object_id}, {', '.join(map(str, np.array(tta_predict).flatten()))}\n")
        if opt.isVal:
            tta_target = targets[object_id][0] # すべて等しい値
            loss = (tta_target-tta_predict)**2
            losses.append(loss)


if opt.isVal:
    RMSE = np.mean(losses)**.5
    with open(Path(opt.checkpoints_dir)/opt.name/f"epoch{opt.which_epoch}_{opt.df_csv[:-4]}_RMSE.csv", "w") as f:
        f.write(f"{RMSE}\n")
