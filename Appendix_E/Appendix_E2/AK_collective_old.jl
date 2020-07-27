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

## Lower bound for the support of the discount factor of both members of the household
theta0=.975

## Setting-up directory
repdir="C:/Users/vaguiar/Dropbox/ReplicationAK" # To run replication code on a different machine change the path
appname="Appendix_E/E2_collective"
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
const repn=(0,500000)   #repn=(burn,number_simulations)
const dg=T              # dg=degrees of freedom
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
## generation of discount factors for each household member
dlength=1
Random.seed!(123)
darand=rand(dlength,n).*(1-theta0).+theta0
Random.seed!(124)
dbrand=rand(dlength,n).*(1-theta0).+theta0

################################################################################
## Initializing

Random.seed!(123)
gammav0=zeros(dg)



################################################################################
## moments
## Moment: my function
include(rootdir*"/cpufunctions/myfun_collective.jl")

## warmstart
darandsim=darand[1,:]
dbrandsim=dbrand[1,:]
include(rootdir*"/cpufunctions/warm_start_searchdelta_justcvexcollective.jl") ##Success is a value of Zero here.
print("warm start ready!")

## chain generation with CUDA
include(rootdir*"/cudafunctions/cuda_chainfun_collective.jl")

Random.seed!(123)
gchain_collective_cu!(theta0,gammav0,cve,rho,chainM)
print("chain ready!")


###########################################################################3
################################################################################################
## Optimization step in cuda
numblocks = ceil(Int, n/167)
const nfast=20000
Random.seed!(123)
indfast=rand(1:repn[2],nfast)
indfast[1]=1
chainMcu=nothing
GC.gc()
chainMcu=cu(chainM[:,:,indfast])
Random.seed!(123)
include(rootdir*"/cudafunctions/cuda_fastoptim.jl")


###############################################################################
###############################################################################
Random.seed!(123)
# res = bboptimize(objMCcu2; SearchRange = (-10e300,10e300), NumDimensions = 4,MaxTime = 100.0,TraceMode=:silent)
#
#
# minr=best_fitness(res)
# TSMC=2*minr*n
# TSMC
# guessgamma=best_candidate(res)
guessgamma=zeros(dg)

###############################################################################
###############################################################################
## First Optimization
## objMCcu(ones(T)*0,ones(T)*0)
opt=NLopt.Opt(:LN_BOBYQA,dg)
#opt=NLopt.Opt(:LN_NELDERMEAD,dg)
toluser=1e-12
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
#NLopt.srand(123::Integer)
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx

###############################################################################
###############################################################################
## Second Optimization
opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=1e-12
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
#NLopt.srand(122::Integer)
(minf,minx,ret) = NLopt.optimize!(opt, solvegamma)
TSMC=2*minf*n
TSMC
solvegamma=minx


# ###############################################################################
# ###############################################################################
# ## Third Optimization
opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=1e-12
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
#NLopt.srand(123::Integer)
(minf,minx,ret) = NLopt.optimize!(opt, solvegamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
#
# ###############################################################################
# ###############################################################################
# ## Fourth Optimization
opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=1e-12
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
#NLopt.srand(124::Integer)
(minf,minx,ret) = NLopt.optimize!(opt, solvegamma)
TSMC=2*minf*n
TSMC
solvegamma=minx

###############################################################################
###############################################################################
## Fifth Optimization
#NLopt.srand(125::Integer)
(minf,minx,ret) = NLopt.optimize!(opt, solvegamma)
TSMC=2*minf*n
TSMC
solvegamma=minx

###############################################################################
###############################################################################
## Sixth Optimization
#NLopt.srand(126::Integer)
(minf,minx,ret) = NLopt.optimize!(opt, solvegamma)
TSMC=2*minf*n
TSMC
solvegamma=minx

###############################################################################
###############################################################################
## Sixth Optimization
#NLopt.srand(127::Integer)
(minf,minx,ret) = NLopt.optimize!(opt, solvegamma)
TSMC=2*minf*n
TSMC
solvegamma=minx

###############################################################################
###############################################################################
## Seventh Optimization
#NLopt.srand(128::Integer)
(minf,minx,ret) = NLopt.optimize!(opt, solvegamma)
TSMC=2*minf*n
TSMC
solvegamma=minx

###############################################################################
###############################################################################
## Eight Optimization
#NLopt.srand(129::Integer)
(minf,minx,ret) = NLopt.optimize!(opt, solvegamma)
TSMC=2*minf*n
TSMC
solvegamma=minx

###############################################################################
###############################################################################
## 9th Optimization
#NLopt.srand(130::Integer)
(minf,minx,ret) = NLopt.optimize!(opt, solvegamma)
TSMC=2*minf*n
TSMC
solvegamma=minx


###############################################################################
###############################################################################
## 9th Optimization
#NLopt.srand(131::Integer)
(minf,minx,ret) = NLopt.optimize!(opt, solvegamma)
TSMC=2*minf*n
TSMC
solvegamma=minx


Results1=DataFrame([theta0 TSMC])
names!(Results1,Symbol.(["theta0","TSGMMcueMC"]))
Results1gamma=DataFrame(hcat(solvegamma,solvegamma))
CSV.write(diroutput*"/AK_collective_TS_cuda_$theta0.csv",Results1)
CSV.write(diroutput*"/AK_collective_gamma_cuda_$theta0.csv",Results1gamma)


print(Results1)