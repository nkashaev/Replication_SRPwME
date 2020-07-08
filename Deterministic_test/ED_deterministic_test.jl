#Version Julia "1.1.1"
using LinearAlgebra
using Random
using MathProgBase
using Clp
using DataFrames
using CSV
################################################################################
## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="Deterministic_test"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all"
dirdata=repdir*"/Data_all"
## Parameters and Functions
stepdum= .03
# Testing function
include(rootdir*"/det_test_ED.jl")
# Number of time periods
const T=4
# Number of goods
const K=17
################################################################################
## Data
# Singles
# Sample size
n=185
###############################################################################
#Prices
dum0=CSV.read(dirdata*"/p.csv",datarow=2)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T,K)
@eval p=$dum0

## Consumption
dum0=CSV.read(dirdata*"/cve.csv",datarow=2)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T,K)
@eval cve=$dum0

## Interest rates
dum0=CSV.read(dirdata*"/rv.csv",datarow=2)
dum0=convert(Matrix,dum0[:,:])
@eval rv=$dum0.+1

## Discounted prices
rho=zeros(n,T,K)
for i=1:n, t=1:T
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
end
## Testing singles
rate_singles=det_test_app(rho,cve,stepdum)
################################################################################


## Couples' households
# Sample size
n=2004
###############################################################################
#Prices
dum0=CSV.read(dirdata*"/pcouple.csv",datarow=2)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T,K)
@eval p=$dum0

## Consumption
dum0=CSV.read(dirdata*"/cvecouple.csv",datarow=2)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T,K)
@eval cve=$dum0

## Interest rates
dum0=CSV.read(dirdata*"/rvcouple.csv",datarow=2)
dum0=convert(Matrix,dum0[:,:])
@eval rv=$dum0.+1

## Discounted prices
rho=zeros(n,T,K)
for i=1:n, t=1:T
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
end
## Testing souples
rate_couples=det_test_app(rho,cve,stepdum)
################################################################################


#Combining results
Results=DataFrame(hcat(["Singles";"Couples"],[rate_singles; rate_couples]))
rename!(Results,Symbol.(["Households","RejRate"]))
CSV.write(diroutput*"/FirstApp_deterministic_tests.csv",Results)
