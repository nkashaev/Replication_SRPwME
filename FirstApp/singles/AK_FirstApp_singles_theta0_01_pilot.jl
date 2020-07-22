###Author: Victor H. Aguiar and Nail Kashaev
###email: vhaguiar@gmail.com
## Version: JULIA 1.1.1 (2020-06-28)

################################################################################
## Loading Packages and setting up procesors
using LinearAlgebra
using Random
using MathProgBase
using DataFrames
using CSV
using NLopt
using BlackBoxOptim
## Lower bound for the support of the discount factor
theta0=0.1

## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="FirstApp"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all"
dirdata=repdir*"/Data_all"
################################################################################
## data size
# Sample size
const n=185
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
include(rootdir*"/singles/ED_data_load.jl") # Function that loads the data
rho,cve=ED_data_load(dirdata,"singles")

print("load data ready!")
################################################################################
## Initializing
Random.seed!(123)
gammav0=zeros(dg)

################################################################################
## Main functions loading and initialization
################################################################################
## moments
## Moment: my function
include(rootdir*"/cpufunctions/myfun.jl")

## warmstart
include(rootdir*"/cpufunctions/warm_start_searchdelta_justcvex.jl")
print("warm start ready!")

## chain generation with CUDA
include(rootdir*"/cudafunctions/cuda_chainfun.jl")
### Random Seed
Random.seed!(123)
gchaincu!(theta0,gammav0,cve,rho,chainM)
print("chain ready!")

###########################################################################3
################################################################################################
## Optimization step in cuda
numblocks = ceil(Int, n/167)
## Select a random subset of the chain from $eta$
const nfast=20000
Random.seed!(123)
indfast=rand(1:repn[2],nfast)
indfast[1]=1
chainMcu=nothing
GC.gc()
chainMcu=cu(chainM[:,:,indfast])
include(rootdir*"/cudafunctions/cuda_fastoptim.jl")

###############################################################################
###############################################################################
### Seed for the BlackBoxOptim first step of Optimization
Random.seed!(123)
res = bboptimize(objMCcu2; SearchRange = (-10e300,10e300), NumDimensions = 4,MaxTime = 400.0,TraceMode=:silent)

minr=best_fitness(res)
TSMC=2*minr*n
guessgamma=best_candidate(res)


###############################################################################
###############################################################################
## First Optimization
opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=1e-6
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
solvegamma=minx

###############################################################################
###############################################################################
## Second Optimization for refinement

opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=1e-6
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
(minf,minx,ret) = NLopt.optimize!(opt, solvegamma)
TSMC=2*minf*n
solvegamma=minx

##############################################################################
Results1=DataFrame([theta0 TSMC])
names!(Results1,Symbol.(["theta0","TSGMMcueMC"]))
CSV.write(diroutput*"/AK_FirstApp_singles_TS_theta0_$theta0.csv",Results1)

print(Results1)
