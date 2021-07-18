#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
from collections import Counter
from typing import *

"""
train と test に 同じシリーズは含まれないそうなので、art_series_id での GroupKFold はやったほうが良さそう
評価指標は RMSE なものの target は {0, 1, 2, 3} なので、StratifiedGroupKFold でとりあえず良さそうですね
Stratified: 重層化する
GroupKFold: trainとtestに同じgroupが入らないように分ける
例）グループとして人のidがあって同じ人を別の角度から取ったデータセットがある場合、
    trainとtestにそれぞれ同じgroupがあるとリークになりえる。
例2)今回の場合は特にtestにtrainと同じシリーズは含まれないため、この方法で分割をしないと
    validationスコアはpublicLBよりも高くでるはず。

当然 set(train)&set(test) = ∅ になるが、
n_splits回Iterationを回す際に、前回のtrainと今回のtrainで重複がでることはある
これは当たり前で

| 1test  | 2train | 3train |  train = 2train + 3train
| 1train | 2test  | 3train |  train = 1train + 3train
| 1train | 2train | 3test  |  train = 1train + 2train

なのでn_splits=Nの場合
train=(N-1)/N, test = 1/N くらいの割合になる
割り切れない場合はデータ件数が変わる場合があるので
評価指標はデータ件数でスケールさせるように注意
"""


def test_SGKFold():
    """SGKFoldのdocstringの例"""
    X = np.ones((17, 2))
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    cv = StratifiedGroupKFold(n_splits=3)
    for train_idxs, test_idxs in cv.split(X, y, groups):
        print("TRAIN:", groups[train_idxs])
        print("      ", y[train_idxs])
        print(" TEST:", groups[test_idxs])
        print("      ", y[test_idxs])

def calc_ratio(train_df: pd.DataFrame) -> np.ndarray:
    cnt = Counter(train_df["target"])
    ratio = np.array([cnt[i] for i in range(4)], dtype=float)
    ratio /= len(train_df)
    return ratio

def main(K=3, batchsize=64):
    """
    train.csvと同じ形式で {K}fold{0}_train.csv ~ {K}fold{K-1}_test.csv を書き出す
    3foldの時、trainの枚数をbatchsize=64の倍数に合わせる為の処理を追加。
    """
    DATADIR = Path("/data/natsuki/dataset_atmaCup11")
    assert DATADIR.is_dir()
    train_df = pd.read_csv(DATADIR/'train.csv')
    ratio = calc_ratio(train_df)

    X = train_df["object_id"]
    y = train_df["target"]
    cv = StratifiedGroupKFold(n_splits=K)
    groups = train_df["art_series_id"]
    for k, (train_idxs, test_idxs) in enumerate(cv.split(X, y, groups)):
        # 最後concatしたときハイパラ(学習時間等)をK/(K-1)しなくちゃいけないかも
        assert len(train_idxs) + len(test_idxs) == len(train_df)
        assert set(train_idxs) & set(test_idxs) == set()
        assert set(groups[train_idxs]) & set(groups[test_idxs]) == set()
        k_train_df = train_df.iloc[train_idxs]
        k_test_df = train_df.iloc[test_idxs]
        k_ratio = calc_ratio(k_train_df)
        emigrant_target = (k_ratio/ratio).argmax()
        prohibitions = set(k_test_df["art_series_id"])
        # ぜんぶ2624 = 64*41にあわせたい
        if not len(train_idxs)%batchsize == 0:
            for emigrant in train_idxs:
                if not train_df.iloc[emigrant]["target"] == emigrant_target:
                    continue
                if train_df.iloc[emigrant]["art_series_id"] in prohibitions:
                    continue
                break
            else:
                raise ValueError("batchsize is incompatible")
            train_idxs = list(train_idxs)
            test_idxs = list(test_idxs)
            train_idxs.remove(emigrant)
            test_idxs.append(emigrant)
            assert len(train_idxs) + len(test_idxs) == len(train_df)
            assert set(train_idxs) & set(test_idxs) == set()
            assert set(groups[train_idxs]) & set(groups[test_idxs]) == set()
            assert len(train_idxs)%batchsize == 0
            k_train_df = train_df.iloc[train_idxs]
            k_test_df = train_df.iloc[test_idxs]
        print(k, len(train_idxs), len(test_idxs))
            
        k_train_df.to_csv(DATADIR/f"{K}fold{k}_train.csv")
        k_test_df.to_csv(DATADIR/f"{K}fold{k}_test.csv")

class PrimeFact:
    def __init__(self, N):
        """
        N以下の素因数分解
        O(NlogNlogN)
        Smallest Prime Factor
        """
        self.spf = list(range(N+1))
        i = 2
        while i*i <= N:
            if self.spf[i] == i:
                for j in range(i*i, N+1, i):
                    if self.spf[j] == j:
                        self.spf[j] = i
            i += 1
    def __call__(self, n):
        # O(logn)
        factor = []
        while n != 1:
            factor.append(self.spf[n])
            n //= self.spf[n]
        return factor

def calc_batchsize():
    P = PrimeFact(1000)
    SM = 80
    for n in range(1, 30):
        H = 224
        W = 224
        C = 3
        N = int((n*(1<<14)*SM)/(H*W*C))
        print(N, 9856%N, P(N))

if __name__ == "__main__":
    main(K=3)
#    main(K=5)