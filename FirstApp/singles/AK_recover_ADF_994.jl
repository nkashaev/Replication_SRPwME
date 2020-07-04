###Author: Victor H. Aguiar and Nail Kashaev
###email: vhaguiar@gmail.com
## Version: JULIA 1.1.0 (2019-01-21)

################################################################################
## Loading Packages and setting up procesors
using LinearAlgebra
using Random
using MathProgBase
using Clp
using DataFrames
using JuMP
using Ipopt
using CSV
using NLopt
using BlackBoxOptim
## Theta
theta0=0.975
avgdelta=0.994

## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="FirstApp"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all"
dirdata=repdir*"/Data_all"

################################################################################
################################################################################
##
# data size
const T=4
const dg=5
T0=4
# Sample size
const n=185
# Number of goods
const K=17
# repetitions for the simulation
## because the simulations are done using parallel Montecarlo we have 100*nprocs draws.
const repn=(0,500000)

###############################################################################
###############################################################################
## Data
#Prices
dum0=CSV.read(dirdata*"/p.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T0,K)
@eval  const p=$dum0

## Consumption
dum0=CSV.read(dirdata*"/cve.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T0,K)
@eval  cve=$dum0

## Interest rates
dum0=CSV.read(dirdata*"/rv.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
@eval const rv=$dum0.+1

###############################################################################
## Discounted prices
rho=zeros(n,T,K)
for i=1:n
  for t=1:(T0)
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
  end
end
rhoold=rho

## Consumption
cveold=cve
cve=zeros(n,T,K)
cve[:,1:T0,:]=cveold
cve[:,T,:]=cveold[:,T0,:]
cve

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
include(rootdir*"/cpufunctions/myfun_recoverdelta.jl")
## chain generation with CUDA
chainM=zeros(n,dg,repn[2])
include(rootdir*"/cudafunctions/cuda_chainfun_delta.jl")
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
include(rootdir*"/cpufunctions/warm_start_searchdelta_justcvex_delta.jl")
print("warm start ready!")


Random.seed!(123)

gchaincu!(theta0,gammav0,cve,rho,chainM)
print("chain ready!")


###########################################################################3
################################################################################################
## Optimization step in cuda
Random.seed!(123)
indfast=rand(1:repn[2],nfast)
indfast[1]=1
chainMcu[:,:,:]=cu(chainM[:,:,indfast])


###############################################################################
###############################################################################
Random.seed!(123)
res = bboptimize(objMCcu2c; SearchRange = (-10e300,10e300), NumDimensions = dg,MaxTime = 100.0, TraceMode=:silent)


minr=best_fitness(res)
TSMC=2*minr*n
TSMC
guessgamma=best_candidate(res)

###############################################################################
###############################################################################


opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=1e-6
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)

#try 1
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret

#try 2
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret


#try 3
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret

#try 4
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret

#try 5
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret


#try 6
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret


#try 7
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret

#try 8
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret

#############################################################
############################################

TSMC





Results1=DataFrame([theta0 avgdelta TSMC])
names!(Results1,Symbol.(["theta0","ADF","TSGMMcueMC"]))
Results1gamma=DataFrame(hcat(solvegamma,solvegamma))
CSV.write(diroutput*"/AK_singles_ADF_cuda_$avgdelta.csv",Results1)
CSV.write(diroutput*"/results_gamma_ADF_$avgdelta.csv",Results1gamma)
