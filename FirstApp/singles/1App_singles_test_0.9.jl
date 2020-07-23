###Author: Victor H. Aguiar and Nail Kashaev
###email: vhaguiar@gmail.com
## Lower bound for the support of the discount factor
theta0=0.9

## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="FirstApp"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all"
dirdata=repdir*"/Data_all"
################################################################################
include(rootdir*"/singles/1App_singles_test.jl")
println("Hi Victor, I'm done")
