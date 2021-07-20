#!/bin/bash

while :; do
    [ $((n++)) -eq 100 ]&& echo $n && exit
done