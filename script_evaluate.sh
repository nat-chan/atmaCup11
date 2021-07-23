#!/bin/bash
set -eux

name=$1
name=$(basename ${name})
name=${name%.yml}
epoch=$2
fold=$3
gpu_ids=${4:-0}

python3 evalute.py --name=${name}_3fold${fold} --df_csv 3fold0_test.csv --conf=./parameters/${name}.yml --conf2=./parameters/test.yml --which_epoch=${epoch} --isVal