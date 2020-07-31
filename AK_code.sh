#!/bin/bash
cd `dirname $0`

K='1 2 3'
for k in $K
do
julia nail_test.jl $k
done