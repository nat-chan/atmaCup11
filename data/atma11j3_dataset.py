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
import torch.nn.functional as F
from glob import glob
from torchvision import transforms as T
import torch
import random


SIZE = (224, 224)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

class Atma11j3Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser: argparse.Parser, is_train: bool) -> argparse.Parser:
        return parser

    def initialize(self, opt: argparse.Namespace) -> None:
        """
        {object_id}-012345678.jpgみたいなのが入った。dirsを指定する
        データかさ増し用のdirsのリスト["photos", photos_h", ...] を持つ。
        """
        self.opt = opt
        self.root = Path(opt.dataroot)
        self.dirs = opt.dirs.split(",")
        self.image_ids: List[str] = list() # image_paths
        for dir in self.dirs:
            dir += "_j3"
            self.image_ids+=list(map(str,(self.root/dir).glob("*.jpg")))

        if self.opt.isTrain:
            self.target_ids  = list(map(lambda s: s.split("-")[-1][:9], self.image_ids))

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

    def load_target(self, p: str) -> torch.Tensor:
        # tensor
        one_hot = F.one_hot(torch.tensor(list(map(int, p))), num_classes=9)
        return torch.flatten(one_hot).float()


    def __getitem__(self, i: int) -> Dict[str, Any]:
        input_dict = {
            'image': self.load_image(self.image_ids[self.perm[i]]).cuda(),
            'path': self.image_ids[self.perm[i]],
        }
        if self.opt.isTrain:
            input_dict.update({
                'target': self.load_target(self.target_ids[self.perm[i]]).cuda(),
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