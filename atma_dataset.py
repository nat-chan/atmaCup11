#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from torch.utils import data
from torchvision import transforms as T

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

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
                False の時は単に size にリサイズを行います
        """

        self.is_train = is_train
        for k in self.meta_keys:
            if k not in meta_df:
                raise ValueError("meta df must have {}".format(k))

        self.meta_df = meta_df.reset_index(drop=True)
        self.index_to_data = self.meta_df.to_dict(orient="index")

        size = (224, 224)

        additional_items = (
            [T.Resize(size)]
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
                T.RandomResizedCrop(size),
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

if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image
    from glob import  glob
    import os

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

    def to_img_path(object_id):
        return os.path.join(photo_dir, f'{object_id}.jpg')

    def read_image(object_id):
        return Image.open(to_img_path(object_id))

    train_meta_df = train_df[['target', 'object_id']].copy()
    train_meta_df['object_path'] = train_meta_df['object_id'].map(to_img_path)


    dataset = AtmaDataset(meta_df=train_meta_df)
    loader = data.DataLoader(dataset=dataset, batch_size=54, num_workers=4)
    for x_tensor, y in loader:
        break
    print(x_tensor.shape)
    print(y.shape)
