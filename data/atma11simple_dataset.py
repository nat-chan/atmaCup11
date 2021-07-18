#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import *
import configargparse as argparse
import pandas as pd
from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
from pathlib import Path
import util.util as util
from os import path
import torch
from glob import glob
from torchvision import transforms as T
import torch
import random


SIZE = (224, 224)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

class Atma11SimpleDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser: argparse.Parser, is_train: bool) -> argparse.Parser:
        return parser

    def initialize(self, opt: argparse.Namespace) -> None:
        """
        データかさ増し用のdirsのリスト["photos", photos_h", ...] を持つ。
        dirの中に{object_id}.jpgが入っていること前提。
        jigsawのような{object_id}-012345678.jpgみたいなのはglob使うほかない。
        TODO: indexへのアクセスに置換配列を噛ませて、epochをseedに決定論的にshuffleする関数を生やす。
            - dataloaderのshuffleはdefaultでFalseだし、Trueにしたときにresumeが決定的だと確かめられたら別にいいかも
        TODO: transform決め打ちだが、optとget_transformで指定出来たりできるようにする。
        TODO: targetをsubmitionの形に変更する関数をDataSetの責務とするか？
        """
        self.opt = opt
        self.root = Path(opt.dataroot)
        self.df = pd.read_csv(self.root/opt.df_csv)
        self.dirs = opt.dirs.split(",")
        self.image_ids: List[str] = list() # image_paths
        if self.opt.isTrain:
            self.target_ids: List[int] = list()
        for dir in self.dirs:
            for i in range(len(self.df)):
                object_id = self.df.iloc[i]["object_id"]
                image_path = str(self.root/dir/f"{object_id}.jpg")
                self.image_ids.append(image_path)
                if self.opt.isTrain:
                    self.target_ids.append(i)
        self.perm = list(range(len(self.image_ids)))
        self.transformer = T.Compose([
            T.Resize(SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ])

    def load_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        image = image.convert('RGB')
        image_tensor = self.transformer(image)
        return image_tensor

    def load_target(self, i: int) -> torch.Tensor:
        target_tensor = torch.Tensor([self.df.iloc[i]["target"]])
        return target_tensor


    def __getitem__(self, i: int) -> Dict[str, Any]:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_dict = {
            'image': self.load_image(self.image_ids[self.perm[i]]).to(device),
            'path': self.image_ids[self.perm[i]],
        }
        if self.opt.isTrain:
            input_dict.update({
                'target': self.load_target(self.target_ids[self.perm[i]]).to(device),
            })

        return input_dict
    
    def shuffle(self, seed: int) -> None: 
        """
        各epochの前にshuffle(seed=epoch)を呼ぶことで
        deterministicにshuffleできるのでresumeも可能。
        random.Randomの独自のインスタンスを作成しているので
        グローバルの乱数には影響がない。
        """
        random.Random(seed).shuffle(self.perm)

    def __len__(self) -> int:
        return len(self.image_ids)