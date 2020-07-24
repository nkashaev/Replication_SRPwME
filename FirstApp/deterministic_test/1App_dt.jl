using LinearAlgebra
using MathProgBase
using Clp
using DataFrames
using CSV

## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="FirstApp"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all"
dirdata=repdir*"/Data_all"

## Parameters
stepdum= .05 # d in [0.1:stepdum:1]

## Function
include(rootdir*"/cpufunctions/ED_det_test.jl")  # ED deterministic test function
include(rootdir*"/cpufunctions/ED_data_load.jl") # Function that loads the data

## Testing singles
rho,cve=ED_data_load(dirdata,"singles") # Data loading
rate_singles=ED_det_test(rho,cve,stepdum) # Testing

## Testing couples
rho,cve=ED_data_load(dirdata,"couples") # Data loading
rate_couples=ED_det_test(rho,cve,stepdum) # Testing

## Combining rounded results
Results=DataFrame(hcat(["Singles";"Couples"],[round(rate_singles*100,digits=1); round(rate_couples*100,digits=1)]))
rename!(Results,Symbol.(["Households","RejRate"]))
CSV.write(diroutput*"/1App_dt_rr.csv",Results)
