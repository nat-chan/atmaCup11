#!/bin/bash
set -eux

R=/data/natsuki/dataset_atmaCup11/photos
rm -rf ${R}_h ${R}_v ${R}_hv ${R}_r ${R}_hr ${R}_vr ${R}_hvr

cp -r ${R} ${R}_h
find ${R}_h -type f |xargs -P 35 -L 200 mogrify -quality 100 -flop

cp -r ${R} ${R}_v
find ${R}_v -type f |xargs -P 35 -L 200 mogrify -quality 100 -flip

cp -r ${R} ${R}_hv
find ${R}_hv -type f |xargs -P 35 -L 200 mogrify -quality 100 -flop -flip

cp -r ${R} ${R}_r
find ${R}_r -type f |xargs -P 35 -L 200 mogrify -quality 100 -rotate 90

cp -r ${R} ${R}_hr
find ${R}_hr -type f |xargs -P 35 -L 200 mogrify -quality 100 -flop -rotate 90

cp -r ${R} ${R}_vr
find ${R}_vr -type f |xargs -P 35 -L 200 mogrify -quality 100 -flip -rotate 90

cp -r ${R} ${R}_hvr
find ${R}_hvr -type f |xargs -P 35 -L 200 mogrify -quality 100 -flop -flip -rotate 90