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
### This first run only initializes the memory in cuda before the loop.
theta0=0.975
kap=10
##Petrol quantity and petrol price target=10, targetgood=10
#target good
targetgood=10
##price change
target=10
bshare=.05

################################################################################
## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="Appendix_F"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all/Appendix"
dirdata=repdir*"/Data_all"


################################################################################
##
# data size
##seed
const T=5
const dg=5


###############################################################################
## Data
## sample size
#singles
const n=185

## time length of the original data
T0=4
## number of goods
const K=17
# repetitions for the simulation
## because the simulations are done using parallel Montecarlo we have 100*nprocs draws.
const repn=(0,10000)



###############################################################################
## Data
###############################################################################

#Prices
dum0=CSV.read(dirdata*"/p.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T0,K)
@eval  const p=$dum0

## Consumption
dum0=CSV.read(dirdata*"/cve.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T0,K)
@eval  const cve=$dum0

## Interest rates
dum0=CSV.read(dirdata*"/rv.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
@eval const rv=$dum0.+1

## Discounted prices
rho=zeros(n,T,K)
for i=1:n
  for t=1:T0
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
  end
end

###############################################################################
## Data Cleaning, Counterfactual prices

## Scaling up rho by kap and adjusting by  0.06 interest rate
for i=1:n
    rho[i,T,:]=rho[i,T-1,:]/(1+0.06)
    rho[i,T,target]=rho[i,T,target]*kap
end


rhoold=rho

## Set Consumption, we initialize the value of the latent consumption C^*_{T+1} to the value C^_{T0}
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
include(rootdir*"/cpufunctions/myfun_counter.jl")
## chain generation with CUDA
chainM=zeros(n,dg,repn[2])
include(rootdir*"/cudafunctions/cuda_chainfun.jl")
## optimization with CUDA
numblocks = ceil(Int, n/100)
const nfast=10000
Random.seed!(123)
indfast=rand(1:repn[2],nfast)
indfast[1]=1
chainMcu=nothing
GC.gc()

chainMcu=cu(chainM[:,:,indfast])
include(rootdir*"/cudafunctions/cuda_fastoptim_counter.jl")
print("functions loaded!")


## warmstart
include(rootdir*"/cpufunctions/warm_start_searchdelta_justcvex.jl")
print("warm start ready!")


Random.seed!(123)

gchaincu!(theta0,gammav0,cve,rho,chainM)
print("chain ready!")


###########################################################################3
################################################################################################
## Optimization step in cuda
chainMcu[:,:,:]=cu(chainM[:,:,indfast])
include(rootdir*"/cudafunctions/cuda_fastoptim_counter.jl")


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
gammav0=randn(dg)*1000
    #gammav0[:]=gamma1
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret
##try 2

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

#############################################################
############################################

TSMC





Results1=DataFrame([theta0 TSMC])
names!(Results1,Symbol.(["theta0","TSGMMcueMC"]))
Results1gamma=DataFrame(hcat(solvegamma,solvegamma))

Results1

print(Results1)
