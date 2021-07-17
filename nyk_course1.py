#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://www.guruguru.science/competitions/17/discussions/e926a5a6-78fd-4bdf-87ff-d08fbff25a02
# %%
from gensim.models import word2vec, KeyedVectors
from glob import  glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2 # type: ignore

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

# TESTがデカすぎるんだよな
assert len(train_df['object_id'])+len(test_df['object_id']) == 9856
# %%

input_df = technique_df

fig, axes = plt.subplots(figsize=(12, 5), ncols=2)

venn2(subsets=(
    set(train_df['object_id']), set(input_df['object_id'])
), set_labels=('train', 'input'), ax=axes[0])

venn2(subsets=(
    set(test_df['object_id']), set(input_df['object_id'])
), set_labels=('test', 'input'), ax=axes[1])

# material の場合
input_df = material_df

fig, axes = plt.subplots(figsize=(12, 5), ncols=2)

venn2(subsets=(
    set(train_df['object_id']), set(input_df['object_id'])
), set_labels=('train', 'input'), ax=axes[0])

venn2(subsets=(
    set(test_df['object_id']), set(input_df['object_id'])
), set_labels=('test', 'input'), ax=axes[1])

# %% targetの分布
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(data=train_df, x='target', ax=ax)
ax.grid()
print(
train_df['target'].value_counts().sort_index()
)
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxenplot(data=train_df, x='target', y='sorting_date', ax=ax)
ax.grid()
# 数字を正確に知りたい場合は groupby を使いましょう。
print(
train_df.groupby('target')['sorting_date'].agg(['min', 'max', 'median', 'size'])
)
# %% モデルの作成
from PIL import Image
from torchvision import transforms as T
import torch
from torchvision.models import resnet34
from torch import nn
from torch.optim import Adam, AdamW
# https://github.com/davda54/sam  SAM面白そうだけどfirst_stepとsecond_stepが必要そう
# from torchsummary import summary TODO

def to_img_path(object_id):
    return os.path.join(photo_dir, f'{object_id}.jpg')

def read_image(object_id):
    return Image.open(to_img_path(object_id))

# test img
converter = T.Compose([
#    T.RandomVerticalFlip(p=.5),
#    T.RandomHorizontalFlip(p=.5), # TODO こっちはやる
#    T.ColorJitter(brightness=.3, contrast=.5, saturation=[.8, 1.3]),
#    T.ColorJitter(brightness=.5, contrast=.5),
    T.ToTensor()
])
# %%
model = resnet34(pretrained=False)
img = read_image(train_df['object_id'].iat[0])
x = converter(img).unsqueeze(0)
output = model(x)
output.shape

# %%
criterion = nn.MSELoss()

# %%
