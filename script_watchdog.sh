#!/bin/bash
name=$1
name=$(basename ${name})
name=${name%.yml}
itertxt=checkpoints/${name}/iter.txt
interbal=$((10*60))
old=s"$(cat ${itertxt}|sed -n 2p)"
while :;do
    new=s"$(cat ${itertxt}|sed -n 2p)"
    if [ "${old}" != "${new}" ]; then
        echo "${new} detected."
        ./script_evaluate.sh ${name} ${new}
    fi
    old="${new}"
    sleep ${interbal}
done
