## Loading Packages.
using LinearAlgebra
using Random
using MathProgBase
using DataFrames
using CSV
using NLopt
using BlackBoxOptim
################################################################################
##Lower bound for the discount factor.
theta0=0.975
# Number of time periods.
const T=4
# Number of moment conditions.
const dg=5
# Sample size.
const n=185
# Number of goods.
const K=17
## MCMC Chain length.
# Burning is optional.
const repn=(0,500000) # repn=(burn,number_simulations)

###############################################################################
## Data
include(rootdir*"/cpufunctions/ED_data_load.jl") # Function that loads the data
rho,cve=ED_data_load(dirdata,"singles") # Loading the data.
print("data loading is ready!")
################################################################################
## Fixing random seed for the random number generator.
Random.seed!(123)
## Initializing gamma.
gammav0=zeros(dg)
################################################################################
## Main functions loading and initialization
################################################################################
## Moment: g(x,e).
include(rootdir*"/cpufunctions/myfun_recoverdelta.jl")
## Generating MCMC chain.
# Generating the first element of the chain.
# Initializing the chain in memory.
chainM=zeros(n,dg,repn[2])
# Loading functions for the generation of the rest of the chain with CUDA -new packages are loaded here.
include(rootdir*"/cudafunctions/cuda_chainfun_delta.jl")
## optimization with CUDA
# Number of blocks for CUDA kernel execution. Parallelization is among individuals size=n.
numblocks = ceil(Int, n/100)  # 100 is a number we obtained from trial and error to speed up the code.
                              # Check your own GPU card's architecture to get the number of blocks that optimizes your speed.
# Select a random subset of the chain (this is equivalent to a random draw from \eta).
# nfast is the number of draws from eta.
const nfast=20000 #20000 is the largest number we can use given GPU memory restrictions.
# Reinitializing random seed.
Random.seed!(123)
# Generate random index.
indfast=rand(1:repn[2],nfast)
# Keep the first element fixed.
indfast[1]=1
chainMcu=nothing
GC.gc()
# chainMcu memory initialization in CUDA.
chainMcu=cu(chainM[:,:,indfast])
## Loading the objective functions in CUDA.
include(rootdir*"/cudafunctions/cuda_fastoptim.jl")
print("functions are loaded!")

## Generating MCMC chain.
# Generating the first element of the chain.
include(rootdir*"/cpufunctions/warm_start_searchdelta_justcvex_delta.jl")
print("chain initialization is ready!")
# Reloading the random seed.
Random.seed!(123)
# Chain generation.
gchaincu!(theta0,gammav0,cve,rho,chainM)
print("chain is ready!")


################################################################################
## Optimization step in CUDA.
# Reinitializing random seed.
Random.seed!(123)
# Generate random index.
indfast=rand(1:repn[2],nfast)
# Keep the first element fixed.
indfast[1]=1
# Passing the draws from eta to CUDA.
chainMcu[:,:,:]=cu(chainM[:,:,indfast])


################################################################################
## Reinitializing the random seed for the BlackBoxOptim first step of Optimization.
Random.seed!(123)
# Optimizes objMCcu2c, limits are theoretically -Inf, Inf, but BlackBoxOptim requires finite bounds.
res = bboptimize(objMCcu2; SearchRange = (-10e300,10e300), NumDimensions = dg,MaxTime = 100.0, TraceMode=:silent)
# Objective value.
minr=best_fitness(res)
# TS value.
TSMC=2*minr*n
TSMC
# Optimal gamma, to be used as a warmstart for the next optimization step.
guessgamma=best_candidate(res)

###############################################################################
## First Optimization of the second step.
# NLopt calling BOBYQA optimizer.
opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=1e-6 # Tolerance parameter.
# Bounds for gamma.
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
# Setting up the tolerance level.
NLopt.xtol_rel!(opt,toluser)
# Optimizing in NLopt, objMCcu is idenical to objMCcu2c except that it is written as required for NLopt.
NLopt.min_objective!(opt,objMCcu)
# Getting (Objective value, optimal gamma, status of optimizer).
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret
# Refinement 2
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret
# Refinement 3
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret
# Refinement 4
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret
# Refinement 5
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret
# Refinement 6
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret
# Refinement 7
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret
# Refinement 8
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret
