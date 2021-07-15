#!/bin/bash

DATADIR=data
NAME=jigsaw3x3

#rm -rf ${DATADIR}/${NAME}
#mkdir -p ${DATADIR}/${NAME}
#cp ${DATADIR}/photos/* ${DATADIR}/${NAME}
#find ${DATADIR}/${NAME} -type f |xargs -P 35 -L 200 mogrify -crop 3x3@ +repage

#puzzle(){
#  [ $# -eq 0 ] && return
#  object_id=${1%.jpg}
#  for a in `seq 0 8`; do
#    for b in `seq 0 8`; do
#      [ $b -eq $a ] && continue
#      for c in `seq 0 8`; do
#        [ $c -eq $a ] && continue
#        [ $c -eq $b ] && continue
#        convert +append \
#          data/jigsaw3x3/${object_id}-${a}.jpg \
#          data/jigsaw3x3/${object_id}-${b}.jpg \
#          data/jigsaw3x3/${object_id}-${c}.jpg \
#          data/jigsaw3x3/${object_id}-${a}${b}${c}.jpg
#      done
#    done
#  done
#  echo ${object_id}
#}
#
#export -f puzzle

#ls ${DATADIR}/photos | xargs -P 35 -n 1 -I% bash -c 'puzzle %' | tee ${NAME}.log
ls ${DATADIR}/photos | xargs -P 35 -L 200 ./sub_preprocessing_jigsaw3x3.py | tee sub${NAME}.log

# vim:ts=2 sw=2
