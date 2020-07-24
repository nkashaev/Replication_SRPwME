## Tested value of the average discount factor
avgdelta=0.992

## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="FirstApp"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all"
dirdata=repdir*"/Data_all"

## Main
include(rootdir*"/procedures/1App_singles_ADF.jl")
println("Success")
