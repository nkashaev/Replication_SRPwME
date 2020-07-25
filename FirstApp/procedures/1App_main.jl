## Loading Packages
using LinearAlgebra
using Random
using MathProgBase
using DataFrames
using CSV
using NLopt
using BlackBoxOptim
################################################################################
# Number of time periods.
const T=4
# Number of goods.
const K=17
## MCMC Chain length.
# Burning is optional.
const repn=(0,500000)       # repn=(burn,number_simulations)
const dg=T                  # dg= number of moment conditions (degrees of freedom)
chainM=zeros(n,dg,repn[2])  # Initializing MCMC chain

################################################################################
## Data
include(rootdir*"/cpufunctions/ED_data_load.jl")    # Function that loads the data
const rho, cve=ED_data_load(dirdata,household)       # Loading the data
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
include(rootdir*"/cpufunctions/myfun.jl")
## Generating MCMC chain.
# Generating the first element of the chain.
include(rootdir*"/cpufunctions/warm_start_searchdelta_justcvex.jl")
print("chain initialization is ready!")
# Loading functions for the generation of the rest of the chain with CUDA -new packages are loaded here.
include(rootdir*"/cudafunctions/cuda_chainfun.jl")
# Reloading the random seed.
Random.seed!(123)
# Chain generation.
gchaincu!(theta0,gammav0,cve,rho,chainM)
print("chain is ready!")

################################################################################
## Optimization step in CUDA.
# Number of blocks for CUDA kernel execution. Parallelization is among individuals size=n.
numblocks = ceil(Int, n/167)    # 167 is a number we obtained from trial and error to speed up the code.
                                # Check your own GPU card's architecture to get the number of blocks that optimizes your speed.
# Select a random subset of the chain (this is equivalent to a random draw from \eta).
# nfast is the number of draws from eta.
const nfast=20000               #20000 is the largest number we can use given GPU memory restrictions.
# Reinitializing random seed.
Random.seed!(123)
# Generate random index.
indfast=rand(1:repn[2],nfast)
# Keep the first element fixed.
indfast[1]=1
# Memory management.
chainMcu=nothing
# Garbage collection.
GC.gc()
# Passing the draws from eta to CUDA.
chainMcu=cu(chainM[:,:,indfast])
## Loading the objective functions in CUDA.
include(rootdir*"/cudafunctions/cuda_fastoptim.jl")

################################################################################
## Reinitializing the random seed for the BlackBoxOptim first step of Optimization.
Random.seed!(123)
# Optimizes objMCcu2, limits are theoretically -Inf, Inf, but BlackBoxOptim requires finite bounds.
res = bboptimize(objMCcu2; SearchRange = (-10e300,10e300), NumDimensions = 4,MaxTime = 400.0,TraceMode=:silent)
# Objective value.
minr=best_fitness(res)
# TS value.
TSMC=2*minr*n
# Optimal gamma, to be used as a warmstart for the next optimization step.
guessgamma=best_candidate(res)

###############################################################################
## First Optimization of the second step.
# NLopt calling BOBYQA optimizer.
opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=1e-6    # Tolerance parameter.
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
# Rewriting the current optimal gamma for next refinement.
solvegamma=minx

###############################################################################
## Second Optimization for refinement using BOBYQA (see Appendix C).
(minf,minx,ret) = NLopt.optimize!(opt, solvegamma)
TSMC=2*minf*n

##############################################################################
