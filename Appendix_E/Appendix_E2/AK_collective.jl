## Loading Packages
using LinearAlgebra
using Random
using MathProgBase
using Clp
using DataFrames
using CSV
using NLopt
using BlackBoxOptim
using JuMP
using Ipopt
## Lower bound for the support of the discount factor of both members of the household
theta0=.975

## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="FirstApp"
rootdir=repdir*"/Appendix_E/Appendix_E2"
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
const dg=T                  # dg= number of moment conditions (degrees of freedom)
chainM=zeros(n,dg,repn[2])  # Initializing MCMC chain

###############################################################################
## Data
include(rootdir*"/cpufunctions/ED_data_load.jl")    # Function that loads the data
const rho, cve=ED_data_load(dirdata,"couples")       # Loading the data
print("data loading is ready!")
################################################################################
## Generation of discount factors for each household member
dlength=1
Random.seed!(123)
darand=rand(dlength,n).*(1-theta0).+theta0
Random.seed!(124)
dbrand=rand(dlength,n).*(1-theta0).+theta0

################################################################################
## Fixing random seed for the random number generator.
Random.seed!(123)
## Initializing gamma.
gammav0=zeros(dg)
################################################################################
## Main functions loading and initialization
################################################################################
## Moment: g(x,e).
include(rootdir*"/cpufunctions/myfun_collective.jl")
## Generating MCMC chain.
# Generating the first element of the chain.
darandsim=darand[1,:]
dbrandsim=dbrand[1,:]
include(rootdir*"/cpufunctions/warm_start_searchdelta_justcvexcollective.jl") ##Success is a value of Zero here.
print("chain initialization is ready!")
# Loading functions for the generation of the rest of the chain with CUDA -new packages are loaded here.
include(rootdir*"/cudafunctions/cuda_chainfun_collective.jl")
# Reloading the random seed.
Random.seed!(123)
# Chain generation.
gchain_collective_cu!(theta0,gammav0,cve,rho,chainM)
print("chain ready!")

################################################################################################
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
# Reinitializing random seed.
Random.seed!(123)
## Loading the objective functions in CUDA.
include(rootdir*"/cudafunctions/cuda_fastoptim.jl")

###############################################################################
## Obtaining an initial guess for gamma using JuMP.
#  This step is not necessary, however, it substantially speeds up computations of the TS.
# Reinitializing the random seed.
Random.seed!(123)
# Initial guess for gamma
guessgamma=zeros(dg)
# Memory management.
modelm=nothing
# Garbage collection.
GC.gc()
# JuMP model
modelm=JuMP.Model(with_optimizer(Ipopt.Optimizer))
@variable(modelm, -10e300 <= gammaj[1:dg] <= 10e300)
chainMnew=chainM[:,:,indfast]
# The convex objective function below has FOC very close to the original moment conditions.
# Thus, giving reliable initial guess for the main optimization problem.
@NLobjective(modelm, Min, sum(log(1+sum(exp(sum(chainMnew[id,t,j]*gammaj[t] for t in 1:dg)) for j in 1:nfast)/nfast) for id in 1:n)/n )
JuMP.optimize!(modelm)
# Intial guess for gamma
for d=1:dg
    guessgamma[d]=JuMP.value(gammaj[d])
end

###############################################################################
## Final Optimization using guessgamma from the previous step.
# NLopt calling BOBYQA optimizer.
opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=1e-24 # Tolerance parameter.
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

## Saving the Output
Results1=DataFrame([theta0 TSMC])
names!(Results1,Symbol.(["theta0","TS"]))
CSV.write(diroutput*"/AK_collective_TS_cuda_$theta0.csv",Results1)

print(Results1)
