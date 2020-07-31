#!/bin/bash

K='1 2 3'
B='4 5 6'
for k in $K
do
for b in $B
do
julia nail_test.jl $k $b
done
done
