## Lower bound for the support of the discount factor
theta0=1.0

## Sample size
const n=2004
#Type of households
household="couples"

## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="FirstApp"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all"
dirdata=repdir*"/Data_all"

## Main
include(rootdir*"/procedures/1App_main.jl")

## Saving the output
Results1=DataFrame([theta0 TSMC])
names!(Results1,Symbol.(["theta0","TS"]))
CSV.write(diroutput*"/1App_couples_main_$theta0.csv",Results1)
println("Success")
