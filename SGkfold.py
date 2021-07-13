#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from icecream import ic
X = np.ones((17, 2))
y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
cv = StratifiedGroupKFold(n_splits=3)
for train_idxs, test_idxs in cv.split(X, y, groups):
    ic(len(train_idxs), len(test_idxs))