#!/bin/bash

DATADIR=data
NAME=jigsaw3x3

choice300(){
#  [ $# -eq 0 ] && return
txt="a
b
c"
  echo ${txt}
  #echo ${object_id}
}

export -f choice300
choice300

#ls ${DATADIR}/photos | xargs -P 35 -n 1 -I% bash -c 'puzzle %' | tee ${NAME}.log
#ls ${DATADIR}/photos | xargs -P 35 -L 200 ./sub_preprocessing_jigsaw3x3.py | tee sub${NAME}.log

# vim:ts=2 sw=2
