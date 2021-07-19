#!/bin/bash
name=$1
name=$(basename ${name})
name=${name%.yml}
epoch=$2
fold=$3
gpu_ids=$4
[ -z "$gpu_ids" ]&&gpu_ids=0

# モデルとデータセットは parameters/atma11simple_xavier_b64.yml とかで管理(_3fold0)とかついていない奴
export CUDA_VISIBLE_DEVICES=${gpu_ids}
python3 evalute.py --name=${name}_3fold${fold} --df_csv 3fold0_test.csv --conf=./parameters/${name}.yml --conf2=./parameters/val.yml --which_epoch=${epoch}