using LinearAlgebra
using Random
using MathProgBase
using Clp
using DataFrames
using CSV
## Theta
npower=10
stepdum= .05
Resultspower=DataFrame(hcat(ones(npower).*1.0,zeros(npower)))
rename!(Resultspower,Symbol.(["seed","RejRate"]))
################################################################################
## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="Appendix_B"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all/Appendix"
dirdata=repdir*"/Data_all"
################################################################################
## Data
# Singles
# Sample size
n=2000
# Number of time periods
const T=4
# Number of goods
const K=17
##
PP=Array(CSV.read(dirdata*"/pcouple.csv",allowmissing=:none))
RR=Array(CSV.read(dirdata*"/rvcouple.csv",allowmissing=:none))
n0=size(PP,1)
PP=reshape(PP,n0,T,K)

## Discounted prices
RRho=zeros(n0,T,K)
for i=1:n0, t=1:T
    RRho[i,t,:]=PP[i,t,:]/prod(1.0 .+ RR[i,1:t])
end

include(rootdir*"/powerfunctions/dgp_nail.jl")
include(rootdir*"/powerfunctions/det_test.jl")


for ri=1:npower
## DGP1
dlow=0.8
rho, cve=dgp12(ri,dlow,n,RRho)
cve=cve*1e5
# Testing DGP 1
# Rejection Rate
rate=det_test_app(rho,cve,stepdum)
Resultspower[ri,1]=ri
Resultspower[ri,2]=rate
CSV.write(diroutput*"/deter_null_v2_theta0_$dlow._n_$n.csv",Resultspower)
GC.gc()
## DGP2
dlow=1.0
rho, cve=dgp12(ri,dlow,n,RRho)
cve=cve*1e5
# Testing DGP 2
# Rejection Rate
rate=det_test_app(rho,cve,stepdum)
Resultspower[ri,1]=ri
Resultspower[ri,2]=rate
CSV.write(diroutput*"/deter_null_v2_theta0_$dlow._n_$n.csv",Resultspower)
GC.gc()
end
