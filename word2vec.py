#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://www.guruguru.science/competitions/16/discussions/2fafef06-5a26-4d33-b535-a94cc9549ac4/
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import word2vec, KeyedVectors
from tqdm import tqdm
tqdm.pandas() # df.progress_apply を生やす


DATADIR = Path("./data")

t = (DATADIR / "techniques.csv")
assert t.is_file()

techniques = pd.read_csv(DATADIR / "techniques.csv")
materials = pd.read_csv(DATADIR / "materials.csv")

materials.head()

# materials.csv, techniques.csv の中には複数のobject_idが含まれているためobject_idで集約するとnameは要素の系列のようになります。
mat_df = materials.groupby("object_id")["name"].apply(list)
materials

# 単語ベクトル表現の次元数
# 元の語彙数をベースに適当に決めました
model_size = {
    "materials": 20,
    "techniques": 8, # TODO ここ20でもよくね
#    "collection": 3,
#    "material_collection": 20,
    "material_technique": 20,
#    "collection_technique": 10,
#    "material_collection_technique": 25
}

n_iter = 100 # TODO 1000でも余裕

w2v_dfs = []
for df, df_name in zip(
        # TODO 元discussionでは積集合も作ってる 
        [
            materials, techniques,
        ], [
            "materials", "techniques",
        ]):
    df_group = df.groupby("object_id")["name"].apply(list).reset_index()
    # Word2Vecの学習
    w2v_model = word2vec.Word2Vec(
        df_group["name"].values.tolist(),
        vector_size=model_size[df_name], # TODO sizeパラメタが存在しない4.0.1
        min_count=1,
        window=1,
        epochs=n_iter,
    )
    # https://radimrehurek.com/gensim/models/word2vec.html
    # epochs (int, optional) – Number of iterations (epochs) over the corpus. (Formerly: iter)

    # 各文章ごとにそれぞれの単語をベクトル表現に直し、平均をとって文章ベクトルにする
    sentence_vectors = df_group["name"].progress_apply(
        lambda x: np.mean([w2v_model.wv[e] for e in x], axis=0))
    sentence_vectors = np.vstack([x for x in sentence_vectors])
    sentence_vector_df = pd.DataFrame(sentence_vectors,
                                      columns=[f"{df_name}_w2v_{i}"
                                               for i in range(model_size[df_name])])
    sentence_vector_df.index = df_group["object_id"]
    w2v_dfs.append(sentence_vector_df)

    w2v_dfs[0].head()


# TODO どっかにdf, df_nameをリスト化して後から参照できるようにする
w2v_dfs[0].to_pickle(DATADIR / f"materials_{n_iter}.pkl")
w2v_dfs[1].to_pickle(DATADIR / f"techniques_{n_iter}.pkl")

