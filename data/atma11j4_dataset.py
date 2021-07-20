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
from collections import defaultdict

SIZE = (224, 224)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


DUPSIZE = 7

def path2p(image_path: str):
    return image_path.split("-")[-1][:9]

def path2k(image_path: str):
    return image_path.split("-")[0]

class Atma11j4Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser: argparse.Parser, is_train: bool) -> argparse.Parser:
        return parser
    

    def initialize(self, opt: argparse.Namespace) -> None:
        """
        {object_id}-012345678.jpgみたいなのが入った。dirsを指定する
        データかさ増し用のdirsのリスト["photos", photos_h", ...] を持つ。
        """
        self.epoch = 1
        self.opt = opt
        self.root = Path(opt.dataroot)
        self.dirs = opt.dirs.split(",")
        self.image_ids = defaultdict(list) # type: ignore
        dup = defaultdict(list)
        for dir in self.dirs:
            dir += "_j3"
            for path in map(str, (self.root/dir).glob("*.jpg")):
                if path2p(path) == "012345678": continue
                dup[path2k(path)].append(path)
        for k, v in dup.items():
            for i in range(0, len(v)//DUPSIZE):
                self.image_ids[i] += v[i*DUPSIZE:(i+1)*DUPSIZE]
                self.image_ids[i] += [f"{k}-012345678.jpg"]

        self.perm = list(range(len(self)))
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

    def load_target(self, image_path: str) -> torch.Tensor:
        p = path2p(image_path)
        one_hot = F.one_hot(torch.tensor(list(map(int, p))), num_classes=9)
        return torch.flatten(one_hot).float()


    def __getitem__(self, i: int) -> Dict[str, Any]:
        input_dict = {
            'image': self.load_image(self.image_ids[self.epoch-1][self.perm[i]]).cuda(),
            'path': self.image_ids[self.epoch-1][self.perm[i]],
        }
        if self.opt.isTrain:
            input_dict.update({
                'target': self.load_target(self.image_ids[self.epoch-1][self.perm[i]]).cuda(),
            })

        return input_dict
    
    def shuffle(self, seed: int) -> None: 
        """
        各epochの前にshuffle(seed=epoch)を呼ぶことで
        deterministicにshuffleできるのでresumeも可能。
        random.Randomの独自のインスタンスを作成しているので
        グローバルの乱数には影響がない。
        """
        self.epoch = seed
        random.Random(seed).shuffle(self.perm)

    def __len__(self) -> int:
        return len(self.image_ids[0])