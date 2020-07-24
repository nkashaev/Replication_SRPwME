## Loading Packages and setting up procesors
using LinearAlgebra
using Random
using MathProgBase
using DataFrames
using CSV
using NLopt
using BlackBoxOptim
################################################################################
# Number of time periods
const T=4
# Number of goods
const K=17
## Repetitions for the integration step
const repn=(0,500000)       # repn=(burn,number_simulations)
const dg=T                  # dg= number of moment conditions (degrees of freedom)
chainM=zeros(n,dg,repn[2])  # Initializaing MCMC chain

################################################################################
## Data
include(rootdir*"/cpufunctions/ED_data_load.jl")    # Function that loads the data
const rho, cve=ED_data_load(dirdata,household)       # Loading the data
print("load data ready!")
################################################################################
## Initializing the initial value for gamma
Random.seed!(123)
gammav0=zeros(dg)
################################################################################
## Main functions loading and initialization
################################################################################
## Moment: my function
include(rootdir*"/cpufunctions/myfun.jl")

## Generating MCMC chain
# Generating the first element of the chain
include(rootdir*"/cpufunctions/warm_start_searchdelta_justcvex.jl")
print("warm start ready!")
# Generation of the rest of the chain with CUDA
include(rootdir*"/cudafunctions/cuda_chainfun.jl")
Random.seed!(123)
gchaincu!(theta0,gammav0,cve,rho,chainM)
print("chain ready!")

################################################################################
## Optimization step in cuda
numblocks = ceil(Int, n/167)
# Select a random subset of the chain from eta
const nfast=20000
Random.seed!(123)
indfast=rand(1:repn[2],nfast)
indfast[1]=1
chainMcu=nothing
GC.gc()
chainMcu=cu(chainM[:,:,indfast])
include(rootdir*"/cudafunctions/cuda_fastoptim.jl")

################################################################################
## Seed for the BlackBoxOptim first step of Optimization
Random.seed!(123)
res = bboptimize(objMCcu2; SearchRange = (-10e300,10e300), NumDimensions = 4,MaxTime = 400.0,TraceMode=:silent)
minr=best_fitness(res)
TSMC=2*minr*n
guessgamma=best_candidate(res)

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
## Second Optimization for refinement
(minf,minx,ret) = NLopt.optimize!(opt, solvegamma)
TSMC=2*minf*n

##############################################################################
