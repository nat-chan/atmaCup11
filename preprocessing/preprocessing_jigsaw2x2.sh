#!/bin/bash

DATADIR=data
NAME=jigsaw2x2

: << EOF
奇数を分割したときに生じる1px幅の差をswapしたときに
どうなるのか気になる
1pxの白線が入るのであればそれを検出するだけになってしまうし
EOF


#rm -rf ${DATADIR}/${NAME}
mkdir -p ${DATADIR}/${NAME}
cp ${DATADIR}/photos/* ${DATADIR}/${NAME}
find ${DATADIR}/${NAME} -type f |xargs -P 22 -L 400 mogrify -crop 50%x50%

puzzle(){
    [ $# -eq 0 ] && return
    object_id=${1%.jpg}
    for a in `seq 0 3`; do
        for b in `seq 0 3`; do
            [ $a -eq $b ] && continue
            convert +append data/jigsaw2x2/${object_id}-${a}.jpg data/jigsaw2x2/${object_id}-${b}.jpg data/jigsaw2x2/${object_id}-${a}${b}.jpg
        done
    done
    for a in `seq 0 3`; do
        for b in `seq 0 3`; do
            [ $b -eq $a ] && continue
            for c in `seq 0 3`; do
                [ $c -eq $a ] && continue
                [ $c -eq $b ] && continue
                for d in `seq 0 3`; do
                    [ $d -eq $a ] && continue
                    [ $d -eq $b ] && continue
                    [ $d -eq $c ] && continue
                    convert -append data/jigsaw2x2/${object_id}-${a}${b}.jpg data/jigsaw2x2/${object_id}-${c}${d}.jpg data/jigsaw2x2/${object_id}-${a}${b}${c}${d}.jpg
                done
            done
        done
    done
    echo ${object_id}
}

export -f puzzle

ls ${DATADIR}/photos | xargs -P 35 -n 1 -I% bash -c 'puzzle %'

mv ${DATADIR}/${NAME} ${DATADIR}/${NAME}_tmp
mkdir -p ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-0123.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-0132.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-0213.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-0231.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-0312.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-0321.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-1023.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-1032.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-1203.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-1230.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-1302.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-1320.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-2013.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-2031.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-2103.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-2130.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-2301.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-2310.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-3012.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-3021.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-3102.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-3120.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-3201.jpg ${DATADIR}/${NAME}
mv ${DATADIR}/${NAME}_tmp/*-3210.jpg ${DATADIR}/${NAME}
