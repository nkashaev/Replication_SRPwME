## Lower bound for the support of the discount factor
theta0=0.1

## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="FirstApp"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all"
dirdata=repdir*"/Data_all"
################################################################################
include(rootdir*"/couples/1App_couples_test.jl")
println("Hi Victor, I'm done")
