#Version Julia "1.1.1"
using LinearAlgebra
using MathProgBase
using Clp
using DataFrames
using CSV
################################################################################
## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
diroutput=repdir*"/Output_all"
dirdata=repdir*"/Data_all"
################################################################################
## Parameters
stepdum= .03 # d in [0.1:stepdum:1]
## Function
include(repdir*"/Deterministic_test/ED_det_test.jl") # ED deterministic test function
include(repdir*"/Deterministic_test/ED_data_load.jl") # Function that loads the data
################################################################################
## Testing singles
rho,cve=ED_data_load(dirdata,"singles") # Data loading
rate_singles=ED_det_test(rho,cve,stepdum) # Testing

## Testing couples
rho,cve=ED_data_load(dirdata,"couples") # Data loading
rate_couples=det_test_app(rho,cve,stepdum) # Testing

## Combining results
Results=DataFrame(hcat(["Singles";"Couples"],[rate_singles; rate_couples]))
rename!(Results,Symbol.(["Households","RejRate"]))
CSV.write(diroutput*"/FirstApp_deterministic_tests.csv",Results)
