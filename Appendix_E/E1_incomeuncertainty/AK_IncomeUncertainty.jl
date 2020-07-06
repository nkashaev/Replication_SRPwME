###Author: Victor H. Aguiar and Nail Kashaev
###email: vhaguiar@gmail.com
## Version: JULIA 1.1.1 (2020-06-28)

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
using JuMP
using Ipopt
using OptimPack

## Lower bound for the support of the discount factor of both members of the household
theta0=1.0

## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="Appendix_E/E1_incomeuncertainty"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all/Appendix"
dirdata=repdir*"/Data_all"
################################################################################
## data size
# Sample size
const n=2004
# Number of time periods
const T=4
# Number of goods
const K=17
## Repetitions for the integration step
const repn=(0,10000)   #repn=(burn,number_simulations)
const dg=7              # dg=degrees of freedom
chainM=zeros(n,dg,repn[2])

###############################################################################
## Data
#Prices
dum0=CSV.read(dirdata*"/pcouple.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T,K)
@eval  const p=$dum0

## Consumption
dum0=CSV.read(dirdata*"/cvecouple.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T,K)
@eval  const cve=$dum0

## Interest rates
dum0=CSV.read(dirdata*"/rvcouple.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
@eval const rv=$dum0.+1

## Discounted prices
rho=zeros(n,T,K)
for i=1:n
  for t=1:T
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
  end
end

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
include(rootdir*"/cpufunctions/myfun_IU_meandisc.jl")
## chain generation with CUDA
include(rootdir*"/cudafunctions/cuda_chainfun_IU_meansdisc.jl")
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
include(rootdir*"/cpufunctions/warm_start_searchdelta_justcvex_IU.jl")  ## expected value is zero, there may be numerical error
print("warm start ready!")


Random.seed!(123)

Delta=rand(n)*(1-theta0).+theta0
gchaincu!(theta0,gammav0,cve,rho,chainM)
print("chain ready!")


###########################################################################3
################################################################################################
## Optimization step in cuda

chainMnew=chainM[:,:,indfast]
chainM=nothing
GC.gc()
chainMcu=cu(chainMnew)
include(rootdir*"/cudafunctions/cuda_fastoptim_counter.jl")


###############################################################################
###############################################################################
## BlackBox Optimization warm-start
Random.seed!(123)
res = bboptimize(objMCcu2c; SearchRange = (-10e300,10e300), NumDimensions = dg,MaxTime = 1000.0, TraceMode=:silent)


minr=best_fitness(res)
TSMC=2*minr*n
TSMC
guessgamma=best_candidate(res)

if (objMCcu2c(guessgamma)>objMCcu2c(zeros(dg)))
     guessgamma=zeros(dg)
end

## BOBYQA refinement
x=guessgamma
## default parameters as in Powell's 2009 paper
bdl = -Inf
bdu =  Inf
xl=ones(dg).*bdl
xu=ones(dg).*bdu
rhobeg = 0.1
rhoend = 1e-6
n0=length(x)
npt = 2*n0 + 1
maxevalpar=minimum([100*(n0+1),1000])

## OptimPack rewrites x
res=OptimPack.Powell.bobyqa!(objMCcu2c, x, xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=1, maxeval=maxevalpar)

minf=res[[3]][1]
TSMC=2*minf*n
println(TSMC)

##Try 2
res=OptimPack.Powell.bobyqa!(objMCcu2c, x, xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=1, maxeval=maxevalpar)

minf=res[[3]][1]
TSMC=2*minf*n
println(TSMC)

##Try 3
res=OptimPack.Powell.bobyqa!(objMCcu2c, x, xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=1, maxeval=maxevalpar)

minf=res[[3]][1]
TSMC=2*minf*n
println(TSMC)

##Try 4
res=OptimPack.Powell.bobyqa!(objMCcu2c, x, xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=1, maxeval=maxevalpar)
minf=res[[3]][1]
TSMC=2*minf*n
println(TSMC)

##Try 5
res=OptimPack.Powell.bobyqa!(objMCcu2c, x, xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=1, maxeval=maxevalpar)
minf=res[[3]][1]
TSMC=2*minf*n
println(TSMC)

##Try 6
res=OptimPack.Powell.bobyqa!(objMCcu2c, x, xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=1, maxeval=maxevalpar)
minf=res[[3]][1]
TSMC=2*minf*n
println(TSMC)

##Try 7
res=OptimPack.Powell.bobyqa!(objMCcu2c, x, xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=1, maxeval=maxevalpar)
minf=res[[3]][1]
TSMC=2*minf*n
println(TSMC)


##Try 8
res=OptimPack.Powell.bobyqa!(objMCcu2c, x, xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=1, maxeval=maxevalpar)
minf=res[[3]][1]
TSMC=2*minf*n
println(TSMC)

##Try 9
res=OptimPack.Powell.bobyqa!(objMCcu2c, x, xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=1, maxeval=maxevalpar)
minf=res[[3]][1]
TSMC=2*minf*n
println(TSMC)


#############################################################
############################################
## Final refinement

opt=NLopt.Opt(:LN_NELDERMEAD,dg)
toluser=1e-12
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
println(TSMC)

solvegamma=minx
guessgamma=solvegamma
ret

opt=NLopt.Opt(:LN_SBPLX,dg)
toluser=1e-6
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
solvegamma=minx
guessgamma=solvegamma
ret
println(TSMC)




Results1=DataFrame([theta0 TSMC])
names!(Results1,Symbol.(["theta0","TSGMMcueMC"]))
Results1gamma=DataFrame(hcat(solvegamma,solvegamma))

CSV.write(diroutput*"/E1_IU_gamma_cuda_$theta0.chain_$nfast.csv",Results1)
CSV.write(diroutput*"/E1_IU_gamma_cuda_$theta0.chain_$nfast.csv",Results1gamma)

Results1

print(Results1)
