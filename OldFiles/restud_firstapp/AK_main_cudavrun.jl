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
using CSV
using NLopt
using BlackBoxOptim
## Theta
theta0=0.5

################################################################################
## Setting-up directory

computer="office"
appname="restud_firstapp"
if computer=="laptop"
    rootdir="D:/Dropbox/AKsource/AKEDapp/"*appname
end
if computer=="office"
    rootdir="D:/Dropbox/Dropbox/AKsource/AKEDapp/"*appname
end
if computer=="lancemachine"
    rootdir="C:/Users/nkashaev/Dropbox/AKsource/AKEDapp/"*appname
end

if computer=="teslamachine"
    rootdir="C:/Users/vaguiar/Dropbox/AKsource/AKEDapp/"*appname
end


################################################################################
##
# data size
##seed
dataapp="singles"
Random.seed!(12)
## sample size
#singles
if dataapp=="singles"
     const n=185
end

if dataapp=="couples"
     const n=2004
end
## time length
const T=4
## number of goods
const K=17
# repetitions for the simulation
## because the simulations are done using parallel Montecarlo we have repn[2]*n draws.
const repn=(0,500000)

const dg=T
chainM=zeros(n,dg,repn[2])

###########################################



###############################################################################
## Data
## Price data from Adams et al.

if dataapp=="singles"
    dir=rootdir*"/singles"
    dirresults=rootdir*"/singles/results"

    dum0=CSV.read(dir*"/p.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    dum0=reshape(dum0,n,T,K)
    @eval  const p=$dum0
    ## Consumption data from Adams et al.
    dum0=CSV.read(dir*"/cve.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    ##original scale in the dataset
    dum0=reshape(dum0,n,T,K)
    @eval  const cve=$dum0
    #sum(ones(n,T,K).*cve,dims=3)

    ## Interest data from Adams et al.
    dum0=CSV.read(dir*"/rv.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    ## This step is done following the replication code in Adams et al.
    @eval const rv=$dum0.+1


end;

if dataapp=="couples"
    dir=rootdir*"/couples"
    dirresults=rootdir*"/couples/results"
    dum0=CSV.read(dir*"/pcouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    dum0=reshape(dum0,n,T,K)
    @eval const p=$dum0
    # consumption array
    dum0=CSV.read(dir*"/cvecouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    #dum0=reshape(dum0,n,T,K)./1e5
    dum0=reshape(dum0,n,T,K)./1e5
    @eval const cve=$dum0

    # interest rate array
    dum0=CSV.read(dir*"/rvcouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    @eval const rv=$dum0.+1


end;

###############################################################################
## Data Cleaning
 rho=zeros(n,T,K)

## Discounted prices
for i=1:n
  for t=1:T
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
  end
end

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
Random.seed!(123)
res = bboptimize(objMCcu2; SearchRange = (-10e300,10e300), NumDimensions = 4,MaxTime = 400.0,TraceMode=:silent)


minr=best_fitness(res)
TSMC=2*minr*n
TSMC
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
TSMC
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
TSMC
solvegamma=minx

##############################################################################




Results1=DataFrame([theta0 TSMC])
names!(Results1,Symbol.(["theta0","TSGMMcueMC"]))
Results1gamma=DataFrame(hcat(solvegamma,solvegamma))
CSV.write(dirresults*"/finalresults_TS_cuda_$theta0.csv",Results1)
CSV.write(dirresults*"/finalresults_gamma_cuda_$theta0.csv",Results1gamma)

Results1

print(Results1)
