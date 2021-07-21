#!/bin/zsh
eval "$($HOME/miniconda3/bin/conda shell.zsh hook)"
conda activate atmaCup11
cd /home/natsuki/atmaCup11
tmuxinator start -p tmuxinator/fold4567.yml --conf=atma11sortingdate_j4nofreeze
sleep 60
./next.sh