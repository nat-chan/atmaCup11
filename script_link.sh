#!/bin/bash

name=$1
name=$(basename $name)
name=${name%.yml}

rm -f tensorboard
ln -s ./checkpoints/$name tensorboard
