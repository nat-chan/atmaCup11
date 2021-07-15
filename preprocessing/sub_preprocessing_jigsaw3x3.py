#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import permutations
from pathlib import Path
import random
import subprocess
import sys


# 362880 ~ 10**5.5 TODO 再利用しよう 380ms
PERM = [''.join(map(str, p)) for p in permutations(range(9))]
DATADIR = Path("./data")
NAME = "jigsaw3x3"
SAMPLE = 100


# convert -append data/jigsaw2x2/${object_id}-${a}.jpg data/jigsaw2x2/${object_id}-${b}.jpg data/jigsaw2x2/${object_id}-${a}${b}.jpg
for object_id in sys.argv[1:]:
    object_id = object_id.split(".")[0] # 拡張子を取り除く
    for p in random.sample(PERM, SAMPLE):
        subprocess.run([
            "convert",
            "-append",
            f"{DATADIR/NAME/object_id}-{p[0:3]}.jpg",
            f"{DATADIR/NAME/object_id}-{p[3:6]}.jpg",
            f"{DATADIR/NAME/object_id}-{p[6:9]}.jpg",
            f"{DATADIR/NAME/object_id}-{p}.jpg",
        ])
    print(object_id)
