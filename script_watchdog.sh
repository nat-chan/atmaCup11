#!/bin/bash

set -eux

name=$1 #suffixを含まない
suffix=$2
gpu_ids=$3
epoch=${4:-1}
root=/data/natsuki/dataset_atmaCup11/checkpoints/${name}_${suffix}
wait=$((1))
while :;do
    if [ -f "${root}/${epoch}_net_G.pth" ]; then
        if [ "${suffix}" = "all" ]; then
            if [ ${epoch} -eq 5 ] || [ ${epoch} -eq 10 ]; then
                python3 evalute.py --name=${name}_${suffix} --df_csv ${suffix}_test.csv  --conf=./parameters/${name}.yml --conf2=./parameters/test.yml --which_epoch=${epoch} --gpu_ids=${gpu_ids}
                python3 evalute.py --name=${name}_${suffix} --df_csv ${suffix}_train.csv --conf=./parameters/${name}.yml --conf2=./parameters/test.yml  --which_epoch=${epoch} --gpu_ids=${gpu_ids} --isVal
                exit
            fi
        else # k-foldの時
            if [ ${epoch} -eq 5 ] || [ ${epoch} -eq 10 ]; then
                python3 evalute.py --name=${name}_${suffix} --df_csv ${suffix}_train.csv --conf=./parameters/${name}.yml --conf2=./parameters/test.yml  --which_epoch=${epoch} --gpu_ids=${gpu_ids} --isVal
            fi
            python3 evalute.py --name=${name}_${suffix} --df_csv ${suffix}_test.csv  --conf=./parameters/${name}.yml --conf2=./parameters/test.yml  --which_epoch=${epoch} --gpu_ids=${gpu_ids} --isVal
        fi
        ((epoch++))
    fi
    sleep ${wait}
done
