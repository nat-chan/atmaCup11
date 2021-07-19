#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import permutations
from pathlib import Path
import random
import subprocess
import sys
from glob import glob


suf = "_j3"
iden = '012345678'
# 362880 ~ 10**5.5 TODO 再利用しよう 380ms
PERM = {''.join(map(str, p)) for p in permutations(range(9))}
PERM.discard(iden)
SAMPLE = 99

src_dir = sys.argv[1]
dst_dir = f"{src_dir}{suf}"
tmp_dir = f"{src_dir}{suf}_tmp"
subprocess.call(["rm", "-rf", dst_dir])
subprocess.call(["rm", "-rf", tmp_dir])
subprocess.call(["mkdir", "-p", dst_dir])
subprocess.call(["cp", "-r", f"{src_dir}/", tmp_dir])
print("crop start")
subprocess.call(f"find {tmp_dir} -type f |xargs -P 35 -L 200 mogrify -crop 3x3@ +repage", shell=True)
print("crop end")

data = glob(f"{src_dir}/*.jpg")
for t, src_fn in enumerate(data):
    ps = src_fn.split("/")
    src_name = ps[-2]
    dst_name = f"{src_name}{suf}"
    tmp_name = f"{src_name}{suf}_tmp"
    object_id = ps[-1].split(".")[0]
    ps[-1] = f"{object_id}-{iden}.jpg"
    ps[-2] = dst_name
    dst_fn = "/".join(ps)
    ps[-2] = tmp_name
    ps[-1] = object_id
    tmp = "/".join(ps)
    subprocess.call(["cp", src_fn, dst_fn])
    for perm in random.sample(PERM, SAMPLE):
        for i in range(0,9,3):
            subprocess.call([
                "convert",
                "-quality",
                "100",
                "+append",
                f"{tmp}-{perm[i+0]}.jpg",
                f"{tmp}-{perm[i+1]}.jpg",
                f"{tmp}-{perm[i+2]}.jpg",
                f"{tmp}-{perm[i:i+3]}.jpg",
            ])
        subprocess.call([
            "convert",
            "-quality",
            "100",
            "-append",
            f"{tmp}-{perm[0:3]}.jpg",
            f"{tmp}-{perm[3:6]}.jpg",
            f"{tmp}-{perm[6:9]}.jpg",
            f"{tmp}-{perm}.jpg",
        ])
        subprocess.call([
            "mv",
            f"{tmp}-{perm}.jpg",
            dst_dir
        ])
    print(dst_name, t, len(data))