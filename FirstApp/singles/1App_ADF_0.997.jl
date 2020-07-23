################################################################################
## Tested value for the average discount factor
avgdelta=0.997

## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="FirstApp"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all"
dirdata=repdir*"/Data_all"
##
include(rootdir*"/singles/1App_ADF.jl")
println("Hi Victor, I'm done")
