using LinearAlgebra
using Random, Distributions
using MathProgBase
using Clp
using DataFrames
using CSV
## Theta
npower=1000
stepdum= .05
Resultspower1=DataFrame(zeros(npower,2))
rename!(Resultspower1,Symbol.(["seed","RejRate"]))
Resultspower2=DataFrame(zeros(npower,2))
rename!(Resultspower2,Symbol.(["seed","RejRate"]))
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
include(repdir*"/Deterministic_test/det_test_ED.jl")


for ri=1:npower
## DGP1
dlow=0.8
rho, cve=dgp12(ri,dlow,n,RRho)
cve=cve*1e5
# Testing DGP 1
# Rejection Rate
rate=det_test_app(rho,cve,stepdum)
Resultspower1[ri,1]=ri
Resultspower1[ri,2]=rate
CSV.write(diroutput*"/deter_null_theta0_$dlow._n_$n.csv",Resultspower1)
GC.gc()
## DGP2
dlow=1.0
rho, cve=dgp12(ri,dlow,n,RRho)
cve=cve*1e5
# Testing DGP 2
# Rejection Rate
rate=det_test_app(rho,cve,stepdum)
Resultspower2[ri,1]=ri
Resultspower2[ri,2]=rate
CSV.write(diroutput*"/deter_null_theta0_$dlow._n_$n.csv",Resultspower2)
GC.gc()
end
## Combining the results
Results=DataFrame(hcat(["DGP1";"DGP2"],[0.0; 0.0]))
rename!(Results,Symbol.(["DGP","RejRate"]))

Results[1,2]=mean(CSV.read(diroutput*"/deter_null_theta0_0.8._n_2000.csv")[:,2])
Results[2,2]=mean(CSV.read(diroutput*"/deter_null_theta0_1.0._n_2000.csv")[:,2])

CSV.write(diroutput*"/deter_null_average_rejecton_rate.csv",Results)
