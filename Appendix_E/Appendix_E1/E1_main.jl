## Loading Packages and setting up procesors
using LinearAlgebra
using Random
using MathProgBase
using DataFrames
using CSV
using NLopt
using BlackBoxOptim

## Lower bound for the support of the discount factor of both members of the household
theta0=1.0

## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
rootdir=repdir*"/Appendix_E/Appendix_E1"
diroutput=repdir*"/Output_all/Appendix"
dirdata=repdir*"/Data_all"
################################################################################
# Sample size
const n=2004
# Number of time periods
const T=4
# Number of goods
const K=17
## MCMC Chain length.
# Burning is optional.
const repn=(0,500000)       #repn=(burn,number_simulations)
const dg=7                  # dg= number of moment conditions (degrees of freedom)
chainM=zeros(n,dg,repn[2])  # Initializing MCMC chain

###############################################################################
## Data
include(rootdir*"/cpufunctions/ED_data_load.jl")    # Function that loads the data
const rho, cve=ED_data_load(dirdata,"couples")       # Loading the data
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
include(rootdir*"/cpufunctions/myfun_IU_meandisc.jl")
# Loading functions for the generation of the chain with CUDA -new packages are loaded here.
include(rootdir*"/cudafunctions/cuda_chainfun_IU_meansdisc.jl")
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
include(rootdir*"/cudafunctions/cuda_fastoptim_counter.jl")
# Generating the first element of the chain.
include(rootdir*"/cpufunctions/warm_start_searchdelta_justcvex_IU.jl")  ## expected value is zero, there may be numerical error
print("chain initialization is ready!")
# Reloading the random seed.
Random.seed!(123)
# Generating d. Since here theta0=1, d=1 a.s.
Delta=rand(n)*(1-theta0).+theta0
# Chain generation.
gchaincu!(theta0,gammav0,cve,rho,chainM)
print("chain ready!")

## Optimization step in CUDA
chainMnew=chainM[:,:,indfast]
chainM=nothing
GC.gc()
chainMcu=cu(chainMnew)
## Loading the objective functions in CUDA.
include(rootdir*"/cudafunctions/cuda_fastoptim_counter.jl")

################################################################################################
## This initial gamma was the product of a brute force search.
trygamma=[-0.021066491;-0.131420248;-0.176570465;-0.061012596;59.08582226;42.73604072;19.77651024]
Random.seed!(123)
guessgamma=trygamma
# NLopt calling BOBYQA optimizer.
opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=0.0
# Bounds for gamma.
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
# Setting up the tolerance level.
NLopt.xtol_rel!(opt,toluser)
NLopt.xtol_abs!(opt,toluser)
# Optimizing in NLopt, objMCcu is idenical to objMCcu2c except that it is written as required for NLopt.
NLopt.min_objective!(opt,objMCcu)
# Getting (Objective value, optimal gamma, status of optimizer).
(minf,minx,ret) = NLopt.optimize(opt, guessgamma)
TSMC=2*minf*n
println(TSMC)

solvegamma=minx
guessgamma=solvegamma

(minf,minx,ret) = NLopt.optimize(opt, guessgamma)
TSMC=2*minf*n
println(TSMC)

solvegamma=minx
guessgamma=solvegamma

## Saving the Output
Results1=DataFrame([theta0 TSMC])
names!(Results1,Symbol.(["theta0","TS"]))
CSV.write(diroutput*"/E1_TS.csv",Results1)
print("success!")
