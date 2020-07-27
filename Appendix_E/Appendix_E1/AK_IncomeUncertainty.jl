###Author: Victor H. Aguiar and Nail Kashaev
###email: vhaguiar@gmail.com
## Version: JULIA 1.1.1 (2020-06-28)

################################################################################
## Loading Packages and setting up procesors
using LinearAlgebra
using Random
using MathProgBase
using Clp
using DataFrames
using CSV
using NLopt
using BlackBoxOptim
using Optim
#using JuMP
#using Ipopt
#using OptimPack

## Lower bound for the support of the discount factor of both members of the household
theta0=1.0

## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="Appendix_E/E1_incomeuncertainty"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all/Appendix"
dirdata=repdir*"/Data_all"
################################################################################
## data size
# Sample size
const n=2004
# Number of time periods
const T=4
# Number of goods
const K=17
## Repetitions for the integration step
const repn=(0,50000)   #repn=(burn,number_simulations)
const dg=7              # dg=degrees of freedom
chainM=zeros(n,dg,repn[2])

###############################################################################
## Data
#Prices
dum0=CSV.read(dirdata*"/pcouple.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T,K)
@eval  const p=$dum0

## Consumption
dum0=CSV.read(dirdata*"/cvecouple.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T,K)
@eval  const cve=$dum0

## Interest rates
dum0=CSV.read(dirdata*"/rvcouple.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
@eval const rv=$dum0.+1

## Discounted prices
rho=zeros(n,T,K)
for i=1:n
  for t=1:T
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
  end
end

print("load data ready!")



################################################################################
## Initializing

###########################################

Random.seed!(123)
gammav0=zeros(dg)




################################################################################
## Main functions loading and initialization

################################################################################
## moments
## Moment: my function
include(rootdir*"/cpufunctions/myfun_IU_meandisc.jl")
## chain generation with CUDA
include(rootdir*"/cudafunctions/cuda_chainfun_IU_meansdisc.jl")
## optimization with CUDA
numblocks = ceil(Int, n/100)
const nfast=20000
Random.seed!(123)
indfast=rand(1:repn[2],nfast)
indfast[1]=1
chainMcu=nothing
GC.gc()


chainMcu=cu(chainM[:,:,indfast])
include(rootdir*"/cudafunctions/cuda_fastoptim_counter.jl")
print("functions loaded!")


## warmstart
include(rootdir*"/cpufunctions/warm_start_searchdelta_justcvex_IU.jl")  ## expected value is zero, there may be numerical error
print("warm start ready!")


Random.seed!(123)

Delta=rand(n)*(1-theta0).+theta0
gchaincu!(theta0,gammav0,cve,rho,chainM)
print("chain ready!")


###########################################################################3
################################################################################################
## Optimization step in cuda

chainMnew=chainM[:,:,indfast]
chainM=nothing
GC.gc()
chainMcu=cu(chainMnew)
include(rootdir*"/cudafunctions/cuda_fastoptim_counter.jl")

################################################################################################
## This initial gamma was the product of a brute force search
trygamma=[-0.021066491;-0.131420248;-0.176570465;-0.061012596;59.08582226;42.73604072;19.77651024]
Random.seed!(123)
guessgamma=trygamma

opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=0.0
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.xtol_abs!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
(minf,minx,ret) = NLopt.optimize(opt, guessgamma)
TSMC=2*minf*n
println(TSMC)

solvegamma=minx
guessgamma=solvegamma
ret





Results1=DataFrame([theta0 TSMC])
names!(Results1,Symbol.(["theta0","TSGMMcueMC"]))
Results1gamma=DataFrame(hcat(solvegamma,solvegamma))

CSV.write(diroutput*"/E1_IU_TS_cuda_$theta0.chain_$nfast.csv",Results1)
CSV.write(diroutput*"/E1_IU_gamma_cuda_$theta0.chain_$nfast.csv",Results1gamma)

Results1

print(Results1)
