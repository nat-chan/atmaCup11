#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import os
from PIL import Image
from torchvision import transforms as T
import torch
from torchvision.models import resnet34
from torch import nn
from torch.optim import Adam, AdamW
from glob import  glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from torch.utils import data

# %% データの置き場

DATADIR = Path("/data/natsuki/dataset_atmaCup11")
assert DATADIR.is_dir()
train_df = pd.read_csv(DATADIR/'train.csv')
test_df = pd.read_csv(DATADIR/'test.csv')
material_df = pd.read_csv(DATADIR/'materials.csv')
technique_df = pd.read_csv(DATADIR/'techniques.csv')
photo_dir = DATADIR/"photos"
output_dir = DATADIR/"outputs_tutorial#1"
os.makedirs(output_dir, exist_ok=True)
photo_pathes = glob(os.path.join(photo_dir, "*.jpg"))
# %%
class AtmaDataset(data.Dataset):
    """atmaCup用にデータ読み込み等を行なうデータ・セット"""
    object_path_key = "object_path"
    label_key = "target"

    @property
    def meta_keys(self):
        retval = [self.object_path_key]
        if self.is_train:
            retval += [self.label_key]
        return retval

    def __init__(self, meta_df: pd.DataFrame, is_train=True):
        """
        args:
            meta_df: 
                画像へのパスと label 情報が含まれている dataframe
                必ず object_path に画像へのパス, target に正解ラベルが入っている必要があります

            is_train:
                True のとき学習用のデータ拡張を適用します.
                False の時は単に SIZE にリサイズを行います
        """
        self.is_train = is_train
        for k in self.meta_keys:
            if k not in meta_df:
                raise ValueError("meta df must have {}".format(k))
        self.meta_df = meta_df.reset_index(drop=True)
        self.index_to_data = self.meta_df.to_dict(orient="index")
        SIZE = (224, 224)
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]
        additional_items = (
            [T.Resize(SIZE)]
            if not is_train
            else [
                T.RandomGrayscale(p=0.2),
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                T.ColorJitter(
                    brightness=0.3,
                    contrast=0.5,
                    saturation=[0.8, 1.3],
                    hue=[-0.05, 0.05],
                ),
                T.RandomResizedCrop(SIZE),
            ]
        )
        self.transformer = T.Compose(
            [*additional_items, T.ToTensor(), T.Normalize(mean=IMG_MEAN, std=IMG_STD)]
        )

    def __getitem__(self, index):
        data = self.index_to_data[index]
        obj_path, label = data.get(self.object_path_key), data.get(self.label_key, -1)
        img = Image.open(obj_path)
        img = self.transformer(img)
        return img, label

    def __len__(self):
        return len(self.meta_df)

# %% utility
def to_img_path(object_id):
    return os.path.join(photo_dir, f'{object_id}.jpg')

def read_image(object_id):
    return Image.open(to_img_path(object_id))
# read_image(train_df['object_id'].iat[0]) #とかする用途

# %% dataset, loaderのインスタンス化
train_meta_df = train_df[['target', 'object_id']].copy()
train_meta_df['object_path'] = train_meta_df['object_id'].map(to_img_path)
dataset = AtmaDataset(meta_df=train_meta_df)
loader = data.DataLoader(dataset=dataset, batch_size=54, num_workers=4)
# %% trainer
from torch.optim.optimizer import Optimizer
from collections import defaultdict

DEVICE = torch.device(4) #nvidia-smiで空いているところ
def train(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: data.DataLoader
) -> pd.Series:

    # train にすることで model 内の学習時にのみ有効な機構が有効になります (Dropouts Layers、BatchNorm Layers...)
    model.train()

    criterion = nn.MSELoss()

    # ロスの値を保存する用に dict を用意
    metrics = defaultdict(float)
    n_iters = len(train_loader)

    for i, (x_i, y_i) in enumerate(train_loader):
        x_i = x_i.to(DEVICE)
        y_i = y_i.to(DEVICE).reshape(-1, 1).float()

        output = model(x_i)
        loss = criterion(output, y_i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_i = {
            # loss は tensor object なので item をつかって python object に戻す
            "loss": loss.item()
        }
        for k, v in metric_i.items():
            metrics[k] += v

    for k, v in metrics.items():
        metrics[k] /= n_iters

    return pd.Series(metrics).add_prefix("train_")
# %% XXX model
import torch
from torchvision.models import resnet34
from torch import nn
model = resnet34(pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=1, bias=True)

# %% XXX outer
from vivid.utils import timer

n_epochs = 10

# GPU 環境で学習するため変換. この呼び出しは破壊的
model.to(DEVICE)
optimizer = Adam(params=model.parameters(), lr=1e-3)

for epoch in range(1, n_epochs + 1):

    with timer(prefix="train: epoch={}".format(epoch)):
        score_train = train(
            model, optimizer, train_loader=loader
        )
    print(score_train)

# %% XXX val
