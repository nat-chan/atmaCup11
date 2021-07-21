#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import pickle

# %%
opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

predicts = defaultdict(list) #type: ignore
with tqdm(dataloader, dynamic_ncols=True) as pbar:
    for i, data_i in enumerate(pbar):
        object_id = data_i["path"][0].split("/")[-1][:-4]
        predict = model(data_i, mode='inference').cpu().detach().numpy()
        predicts[object_id].append(predict)

with open(Path(opt.checkpoints_dir)/opt.name/f"epoch{opt.which_epoch}_{opt.df_csv[:-4]}_features2.pkl", "wb") as f:
    pickle.dump(predicts, f)