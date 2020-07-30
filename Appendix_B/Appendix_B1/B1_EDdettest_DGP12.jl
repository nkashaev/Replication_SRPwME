using LinearAlgebra
using Random
using MathProgBase
using Clp
using CSV, DataFrames
################################################################################
## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="Appendix_B"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all/Appendix"
dirdata=repdir*"/Data_all"
################################################################################
## Parameters
nrepl=1000 #Number of MC replications
stepdum=.05 # d in [0.1:stepdum:1]
n=2000      #Sample size of the generated sample
## Functions
include(rootdir*"/Appendix_B1/dgp_12.jl") #Functions that generate the data
include(repdir*"/FirstApp/cpufunctions/ED_det_test.jl") # ED deterministic test function
include(repdir*"/FirstApp/cpufunctions/ED_data_load.jl") # Function that loads the data
## Output files
Resultspower1=DataFrame(zeros(nrepl,2))
rename!(Resultspower1,Symbol.(["seed","RejRate"]))
Resultspower2=DataFrame(zeros(nrepl,2))
rename!(Resultspower2,Symbol.(["seed","RejRate"]))
Results=DataFrame(hcat(["DGP1";"DGP2"],[0.0; 0.0]))
rename!(Results,Symbol.(["DGP","AveRejRate"]))

## Data
RRho,CVEt=ED_data_load(dirdata,"couples")

## Simulations
for ri=1:nrepl
    # DGP1
    dlow=0.8
    rho, cve=dgp12(ri,dlow,n,RRho)
    # Testing DGP 1
    rate=ED_det_test(rho,cve,stepdum) # Rejection Rate
    Resultspower1[ri,1]=ri; Resultspower1[ri,2]=rate;
    CSV.write(diroutput*"/B1_EDdettest_rr_dgp1.csv",Resultspower1)
    GC.gc()
    ## DGP2
    dlow=1.0
    rho, cve=dgp12(ri,dlow,n,RRho)
    # Testing DGP 2
    rate=ED_det_test(rho,cve,stepdum) # Rejection Rate
    Resultspower2[ri,1]=ri; Resultspower2[ri,2]=rate;
    CSV.write(diroutput*"/B1_EDdettest_rr_dgp2.csv",Resultspower2)
    GC.gc()
end
## Combining the results
Results[1,2]=sum(Resultspower1[:,2])/nrepl
Results[2,2]=sum(Resultspower2[:,2])/nrepl

CSV.write(diroutput*"/B1_EDdettest_arr_dgp12.csv",Results)
