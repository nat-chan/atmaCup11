#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://www.guruguru.science/competitions/16/discussions/2fafef06-5a26-4d33-b535-a94cc9549ac4/
# %%
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import word2vec, KeyedVectors
from typing import *
from tqdm import tqdm
tqdm.pandas() # df.progress_apply を生やす
from typing import *
import torch
# %%


DATADIR = Path("/data/natsuki/dataset_atmaCup11")
assert DATADIR.is_dir()

t = (DATADIR / "techniques.csv")
assert t.is_file()

materials = pd.read_csv(DATADIR / "materials.csv")
techniques = pd.read_csv(DATADIR / "techniques.csv")

train_df = pd.read_csv(DATADIR/'train.csv')
test_df = pd.read_csv(DATADIR/'test.csv')
all_train_df = pd.read_csv(DATADIR/'all_train.csv')

materials_set, techniques_set = set(materials["name"]), set(techniques["name"])
print(materials_set & techniques_set) # {'pencil', 'chalk'}

# %% XXX materialsの方を "pencil" ->  "pencil (material)" に変更
materials["name"] = materials["name"].apply(lambda x: f"{x} (material)" if x in {'pencil', 'chalk'} else  x)

materials_set, techniques_set = set(materials["name"]), set(techniques["name"])
assert materials_set & techniques_set == set()
# %%
materials_techniques = pd.concat([materials, techniques])
materials_techniques_df = materials_techniques.groupby("object_id")["name"].apply(list).reset_index()

# %%
materials_df = materials.groupby("object_id")["name"].apply(list).reset_index()
techniques_df = materials.groupby("object_id")["name"].apply(list).reset_index()
print("train.csv", set(train_df["object_id"])-set(materials_df["object_id"]))
print("all_train.csv", set(all_train_df["object_id"])-set(materials_df["object_id"]))
# %%


# %%
def w2v(
        df_group: pd.DataFrame,
        df_name: str,
        vector_size: int = 20,
        epochs: int = 100,
    ) -> pd.DataFrame:
    """
    https://radimrehurek.com/gensim/models/word2vec.html
    epochs (int, optional) – Number of iterations (epochs) over the corpus. (Formerly: iter)
    materials.csv, techniques.csv の中には複数のobject_idが含まれているため
    object_idで集約するとnameは要素の系列のようになります。
    """
    w2v_model = word2vec.Word2Vec(
        df_group["name"].values.tolist(),
        vector_size=vector_size, # TODO sizeパラメタが存在しない4.0.1
        min_count=1,
        window=1,
        epochs=epochs,
    )

    # 各文章ごとにそれぞれの単語をベクトル表現に直し、平均をとって文章ベクトルにする
    sentence_vectors = df_group["name"].progress_apply(
        lambda x: np.mean([w2v_model.wv[e] for e in x], axis=0))
    sentence_vectors = np.vstack([x for x in sentence_vectors])
    sentence_vector_df = pd.DataFrame(
        sentence_vectors,
        columns=[
            f"{df_name}_w2v_{i}"
            for i in range(vector_size)
        ]
    )
    sentence_vector_df.index = df_group["object_id"]
    return sentence_vector_df

# %%
for epochs in [100, 1000]:
    name = "materialstechniques"
    sentence_vector_df = w2v(materials_techniques_df, name, epochs=epochs)
    sentence_vector_df.to_pickle(DATADIR/f"{name}_{epochs}.pkl")
# %%
self = type(str(), tuple(), dict())
self.outer_df = pd.read_pickle( DATADIR / f"materialstechniques_100.pkl" )
self.outer_df.head()
# %%
self.df = train_df
# %%
# %%
i = 0
train_df.iloc[i]["object_id"]

torch.tensor(self.outer_df.loc[self.df.iloc[i]["object_id"]].values)

# %%
