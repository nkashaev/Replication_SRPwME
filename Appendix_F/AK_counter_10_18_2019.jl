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
theta0=0.975
@everywhere kap=1.01
@everywhere target=10
@everywhere targetgood=10
@everywhere bshare=.05

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
@everywhere const T=5
@everywhere  const dg=5


###############################################################################
## Data
## Price data from Adams et al.

##seed
dataapp="singles"
@everywhere Random.seed!(12)
## sample size
#singles
if dataapp=="singles"
    @everywhere  const n=185
end

if dataapp=="couples"
    @everywhere  const n=2004
end
## time length of the original data
@everywhere T0=4
## number of goods
@everywhere const K=17
# repetitions for the simulation
## because the simulations are done using parallel Montecarlo we have 100*nprocs draws.
@everywhere const repn=(0,10000)


## number of proccesors
const nprocs0=nprocsdum+1

###########################################


###############################################################################
## Data
## Price data from Adams et al.

if dataapp=="singles"
    dir=rootdir*"/singles"
    dirresults=rootdir*"/singles/results"

    dum0=CSV.read(dir*"/p.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    dum0=reshape(dum0,n,T0,K)
    @eval @everywhere  const p=$dum0
    ## Consumption data from Adams et al.
    dum0=CSV.read(dir*"/cve.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    ##original scale in the dataset
    dum0=reshape(dum0,n,T0,K)
    @eval @everywhere   cve=$dum0

    ## Interest data from Adams et al.
    dum0=CSV.read(dir*"/rv.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    ## This step is done following the replication code in Adams et al.
    @eval @everywhere const rv=$dum0.+1


end;

if dataapp=="couples"
    dir=rootdir*"/couples"
    dirresults=rootdir*"/couples/results"
    dum0=CSV.read(dir*"/pcouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    dum0=reshape(dum0,n,T0,K)
    @eval @everywhere const p=$dum0
    # consumption array
    dum0=CSV.read(dir*"/cvecouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    #dum0=reshape(dum0,n,T,K)./1e5
    dum0=reshape(dum0,n,T0,K)./1e5
    @eval @everywhere const cve=$dum0

    # interest rate array
    dum0=CSV.read(dir*"/rvcouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    @eval @everywhere const rv=$dum0.+1


end;



###############################################################################
## Data Cleaning, Counterfactual prices
@everywhere  rho=zeros(n,T,K)

## Discounted prices
@everywhere for i=1:n
  for t=1:(T0)
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
  end
end
## Scaling up rho by kap
@everywhere for i=1:n
    rho[i,T,:]=rho[i,T-1,:]*($theta0)^(T)
    rho[i,T,target]=rho[i,T,target]*kap
end



rhoold=rho

## Set Consumption
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
include(rootdir*"/restud_counter/cpufunctions/myfun_counter.jl")
## chain generation with CUDA
chainM=zeros(n,dg,repn[2])
include(rootdir*"/restud_counter/cudafunctions/cuda_chainfun.jl")
## optimization with CUDA
numblocks = ceil(Int, n/100)
@everywhere global nfast=10000
Random.seed!(123)
indfast=rand(1:repn[2],nfast)
indfast[1]=1
#indfast=repn[2]-nfast+1:repn[2]
#indfast=1:nfast
chainMcu=nothing
GC.gc()

chainMcu=cu(chainM[:,:,indfast])
include(rootdir*"/restud_counter/cudafunctions/cuda_fastoptim_counter.jl")
print("functions loaded!")


## warmstart
include(rootdir*"/restud_counter/cpufunctions/warm_start_searchdelta_justcvex.jl")
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
#res = bboptimize(objMCcu2; SearchRange = (-10e300,10e300), NumDimensions = 4,MaxTime = 100.0, TraceMode=:silent)


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

TSMC





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
