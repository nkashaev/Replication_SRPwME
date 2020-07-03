###Author: Victor H. Aguiar and Nail Kashaev
###email: vhaguiar@gmail.com
## Version: JULIA 1.1.0 (2019-01-21)

################################################################################
## Loading Packages and setting up procesors
using Distributed
count = 0
nprocsdum=1
addprocs(nprocsdum)
@everywhere using LinearAlgebra
@everywhere using Random
@everywhere using MathProgBase
@everywhere using Clp
@everywhere using DataFrames
using JuMP
using Ipopt
using CSV
using NLopt
using BlackBoxOptim
using Sobol
## Theta
beta=1.0
theta0=.1

################################################################################
## Setting-up directory
#rootdir="D:/Dropbox/AKsource/AKEDapp"
computer="lancemachine"
if computer=="laptop"
    rootdir="D:/Dropbox/AKsource/AKEDapp"
end
if computer=="office"
    rootdir="D:/Dropbox/Dropbox/AKsource/AKEDapp"
end
if computer=="lancemachine"
    rootdir="C:/Users/nkashaev/Dropbox/AKsource/AKEDapp"
end


################################################################################
##
# data size
##seed
dataapp="couples"
@everywhere Random.seed!(12)
## sample size


if dataapp=="couples"
    @everywhere  const n=2000
end
## time length
@everywhere const T=4
## number of goods
@everywhere const K=17
# repetitions for the simulation
## because the simulations are done using parallel Montecarlo we have 100*nprocs draws.
@everywhere const repn=(0,10000)

chainM=zeros(n,T,repn[2])
## number of proccesors
const nprocs0=nprocsdum+1

###########################################
@everywhere const dg=T


###############################################################################
## Data
## Price data from Adams et al.

if dataapp=="couples"
    dir=rootdir*"/couples"
    dirresults=rootdir*"/power/results"
    dum0=CSV.read(dir*"/pcouple.csv",allowmissing=:none)
    indsim=rand(1:2004,n)
    dum0=convert(Matrix,dum0[indsim,:])
    dum0=reshape(dum0,n,T,K)
    @eval @everywhere const p=$dum0
    # consumption array
    dum0=CSV.read(dir*"/cvecouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[indsim,:])
    #dum0=reshape(dum0,n,T,K)./1e5
    dum0=reshape(dum0,n,T,K)
    @eval @everywhere const cve=$dum0

    # interest rate array
    dum0=CSV.read(dir*"/rvcouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[indsim,:])
    @eval @everywhere const rv=$dum0.+1


end;


###############################################################################
## Data Cleaning
@everywhere  rho=zeros(n,T,K)

## Discounted prices
@everywhere for i=1:n
  for t=1:T
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
  end
end

rhoold=rho

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
include(rootdir*"/restud_apppower/cpufunctions/myfun.jl")
## chain generation with CUDA
include(rootdir*"/restud_apppower/cudafunctions/cuda_chainfun.jl")
## optimization with CUDA
numblocks = ceil(Int, n/100)
nfast=10000
Random.seed!(123)
indfast=rand(1:repn[2],nfast)
indfast[1]=1
#indfast=repn[2]-nfast+1:repn[2]
#indfast=1:nfast
chainMcu=nothing
GC.gc()
chainMcu=cu(chainM[:,:,indfast])
include(rootdir*"/restud_apppower/cudafunctions/cuda_fastoptim.jl")
print("functions loaded!")


#############################################################################
## Data generation
##Hyperbolic
#include(rootdir*"/restud_apppower/powerfunctions/hyperbolicgenerator_simulations_2019_9_26.jl")

## Collective
include(rootdir*"/restud_apppower/powerfunctions/couples_generator_simulations_2019_9_28.jl")

## warmstart
include(rootdir*"/restud_apppower/cpufunctions/warm_start_searchdelta_justcvex.jl")
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
res = bboptimize(objMCcu2; SearchRange = (-10e300,10e300), NumDimensions = 4,MaxTime = 100.0, TraceMode=:silent)
#res = bboptimize(objMCcu2; SearchRange = (-10e5,10e5), NumDimensions = 4,MaxTime = 100.0,Method=:probabilistic_descent)


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
gammav0=randn(dg)*1000
    #gammav0[:]=gamma1
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret
##try 2
if (TSMC>=9)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
end
#try 3
if (TSMC>=9)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
end

#############################################################
############################################







Results1=DataFrame([theta0 TSMC])
names!(Results1,Symbol.(["theta0","TSGMMcueMC"]))
#names!(Results1,Symbol.(["theta0","TSGMMnovar","TSGMMcue","TSGMMcueMC"]))
Results1gamma=DataFrame(hcat(solvegamma,solvegamma))
#CSV.write(dirresults*"/chainM50K.csv",DataFrame(reshape(chainM,n*T,repn[2])))
#CSV.write()
#CSV.write(dirresults*"/results_TS_cuda_$theta0.csv",Results1)
#CSV.write(dirresults*"/results_gamma_cuda_$theta0.csv",Results1gamma)

Results1

print(Results1)
