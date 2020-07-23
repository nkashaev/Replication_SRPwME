## Lower bound for the support of the discount factor
theta0=0.9
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
################################################################################
include(rootdir*"/cpufunctions/1App_main.jl")

##Saving the output
Results1=DataFrame([theta0 TSMC])
names!(Results1,Symbol.(["theta0","TS"]))
CSV.write(diroutput*"/AK_FirstApp_couples_TS_theta0_$theta0.csv",Results1)
println("Hi Victor, I'm done")
