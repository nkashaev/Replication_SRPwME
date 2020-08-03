## Loading Packages
using LinearAlgebra
using Random
using MathProgBase
using DataFrames
using CSV
using NLopt
using BlackBoxOptim
#
using CuArrays
using CuArrays.CURAND
using CUDAnative
using CUDAdrv
#
using Convex
using ECOS

using SoftGlobalScope
###############################################################################
## Counterfactual
# Number of time periods.
const T=5
const dg=5 # dg= number of moment conditions (degrees of freedom)
###############################################################################
## Data
## Sample size.
const n=185
# Number of original time periods.
const T0=4
# Number of goods.
const K=17
## MCMC Chain length.
# Burning is optional.
const repn=(0,10000) #repn=(burn,number_simulations)
chainM=zeros(n,dg,repn[2]) # Initializing MCMC chain
# nfast is the number of draws from eta.
const nfast=10000
# Reinitializing random seed.
Random.seed!(123)
# Generate random index.
indfast=rand(1:repn[2],nfast)
# Keep the first element fixed.
indfast[1]=1
# Passing the draws from eta to CUDA.
chainMcu=cu(chainM[:,:,indfast])
# Lower bound for d.
theta0=0.975
## Target good
# 10 petrol
targetgood=10
# Price change
target=10

const rho=zeros(n,T,K)
const cve=zeros(n,T,K)

# Values for kappa.
kapvec=[1.0 1.01 1.02 1.03 1.04 1.05 1.06 1.07 1.08 1.09 1.10]
nkap=length(kapvec)
## Budget share start.
startit=.04
## Budget share end.
endit=0.06
## Step of the search.
step=.0005
gridvec=collect(startit:step:endit)
npower=length(gridvec)
Resultspower=DataFrame(hcat(ones(npower),zeros(npower)))
names!(Resultspower,Symbol.(["bshare","TSGMMcueMC"]))
## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="Appendix_F"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all/Appendix"
dirdata=repdir*"/Data_all"


###############################################################################
## Data
###############################################################################
#Prices
dum0=CSV.read(dirdata*"/p.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T0,K)
@eval  p=$dum0

## Consumption
dum0=CSV.read(dirdata*"/cve.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T0,K)
@eval  cvetemp=$dum0

## Interest rates
dum0=CSV.read(dirdata*"/rv.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
@eval rv=$dum0.+1

################################################################################
## Main functions loading and initialization
################################################################################
## Moment: g(x,e).
include(rootdir*"/cpufunctions/myfun_counter.jl")
# Chain generation with CUDA.
chainM[:,:,:]=zeros(n,dg,repn[2])
include(rootdir*"/cudafunctions/cuda_chainfun.jl")
# Optimization with CUDA..
numblocks = ceil(Int, n/100)
chainMcu[:,:,:]=cu(chainM[:,:,indfast])
include(rootdir*"/cudafunctions/cuda_fastoptim_counter.jl")
print("functions are loaded!")


### ARGS -- passing arguments from the command line (see loop.ps1 file)
ki=parse(Int32,ARGS[1])
ri=parse(Int32,ARGS[2])
## Continue
kap=kapvec[ki]
bshare=gridvec[ri]
## Discounted prices
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

## Set Consumption. We initialize the value of the latent consumption C^*_{T+1} to the value C^_{T0}
cve[:,1:T0,:]=cvetemp
cve[:,T,:]=cvetemp[:,T0,:]
cve
print("load data ready!")

################################################################################
## Fixing random seed for the random number generator.
Random.seed!(123)
## Initializing gamma.
gammav0=zeros(dg)
## Generating the initial element of the MCMC chain
deltavec=theta0<1 ? [0 .5  1]*(1-theta0).+theta0 : [1]
ndelta=length(deltavec)
Delta=zeros(n)
Deltatemp=zeros(n)
W=ones(n,T,K)
cvesim=zeros(n,T,K)
vsim=zeros(n,T)
optimval=ones(n,ndelta+1)*10000
aiverify2=zeros(n,T,T)
v=Variable(T, Positive())
c=Variable(T,K,Positive())
P=I+zeros(1,1)

for dt=2:ndelta+1
    for id=1:n
        Deltatemp[id]=deltavec[dt-1]
        modvex=minimize(quadform(rho[id,1,:]'*(c[1,:]'-cve[id,1,:]),P)+quadform(rho[id,2,:]'*(c[2,:]'-cve[id,2,:]),P)+quadform(rho[id,3,:]'*(c[3,:]'-cve[id,3,:]),P)+quadform(rho[id,4,:]'*(c[4,:]'-cve[id,4,:]),P))
        for t=1:T
            for s=1:T
                modvex.constraints+=v[t]-v[s]-Deltatemp[id]^(-(t-1))*rho[id,t,:]'*(c[t,:]'-c[s,:]')>=0
            end
        end
        solve!(modvex,ECOSSolver(verbose=false))

        optimval[id,dt]=modvex.optval

        aiverify=zeros(n,T,T)

        Delta[id]=Deltatemp[id]
        for i=1:T
            vsim[id,i]=v.value[i]
            for j=1:K
                cvesim[id,i,j]=c.value[i,j]
            end
        end

        for t=1:T
            for s=1:T
                aiverify2[id,t,s]=vsim[id,t]-vsim[id,s]-Delta[id]^(-(t-1))*rho[id,t,:]'*(cvesim[id,t,:]-cvesim[id,s,:])
            end
        end
end
    modvex=nothing
    GC.gc()
end

minimum(aiverify2)
print("warm start ready!")

Random.seed!(123)
print(bshare)
gchaincu!(theta0,gammav0,cve,rho,chainM,Delta,vsim,cvesim,W,bshare)
print("chain ready!")

###########################################################################3
################################################################################################
## Optimization step in cuda
Random.seed!(123*ki*ri)
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
(minf,minx,ret) = NLopt.optimize(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret
##try 2
(minf,minx,ret) = NLopt.optimize(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret
#try 3
(minf,minx,ret) = NLopt.optimize(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret
########

if (ri>1)
    Resultspower[:,:]=CSV.read(diroutput*"/F_$kap._$theta0.csv",datarow=2)
end

Resultspower[ri,2]=TSMC
Resultspower[ri,1]=bshare


CSV.write(diroutput*"/F_$kap._$theta0.csv",Resultspower)
println("success!")
GC.gc()
