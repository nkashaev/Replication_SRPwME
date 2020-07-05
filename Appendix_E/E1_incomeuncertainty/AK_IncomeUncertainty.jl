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
const repn=(0,1500000)   #repn=(burn,number_simulations)
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
const nfast=200000
Random.seed!(123)
indfast=rand(1:repn[2],nfast)
indfast[1]=1
#indfast=repn[2]-nfast+1:repn[2]
#indfast=1:nfast
chainMcu=nothing
GC.gc()


chainMcu=cu(chainM[:,:,indfast])
include(rootdir*"/cudafunctions/cuda_fastoptim_counter.jl")
print("functions loaded!")


## warmstart
include(rootdir*"/cpufunctions/warm_start_searchdelta_justcvex_IU.jl")
print("warm start ready!")


Random.seed!(123)

Delta=rand(n)*(1-theta0).+theta0
gchaincu!(theta0,gammav0,cve,rho,chainM)
print("chain ready!")


###########################################################################3
################################################################################################
## Optimization step in cuda

Random.seed!(123)
indfast=rand(1:repn[2],nfast)
indfast[1]=1
chainMcu=nothing
GC.gc()
chainMnew=chainM[:,:,indfast]
chainM=nothing
GC.gc()
chainMcu=cu(chainMnew)
include(rootdir*"/cudafunctions/cuda_fastoptim_counter.jl")


###############################################################################
###############################################################################

modelm=nothing
GC.gc()
modelm=JuMP.Model(with_optimizer(Ipopt.Optimizer))
@variable(modelm, -10e300 <= gammaj[1:dg] <= 10e300)

@NLobjective(modelm, Min, sum(log(1+sum(exp(sum(chainMnew[id,t,j]*gammaj[t] for t in 1:dg)) for j in 1:nfast)/nfast) for id in 1:n)/n )



JuMP.optimize!(modelm)


guessgamma=zeros(dg)
for d=1:dg
    guessgamma[d]=JuMP.value(gammaj[d])
end

# Random.seed!(123)
# res = bboptimize(objMCcu2c; SearchRange = (-10e300,10e300), NumDimensions = dg,MaxTime = 100.0, TraceMode=:silent)
# #res = bboptimize(objMCcu2; SearchRange = (-10e300,10e300), NumDimensions = 4,MaxTime = 100.0, TraceMode=:silent)


# minr=best_fitness(res)
# TSMC=2*minr*n
# TSMC
# guessgamma=best_candidate(res)

# if (TSMC>1000)
#     guessgamma=zeros(dg)
# end

###############################################################################
###############################################################################
guessgamma=[-37.91552242
34.32517357
1.157511448
-0.861352236
-6.133957924
-6.699938902
-4.957071496
]

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
if (TSMC>=0)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
    TSMC
end
#try 3
if (TSMC>=0)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
    TSMC
end

if (TSMC>=0)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
    TSMC
end

if (TSMC>=0)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
    TSMC
end

if (TSMC>=0)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
    TSMC
end

if (TSMC>=0)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
    TSMC
end

if (TSMC>=0)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
    TSMC
end
if (TSMC>=0)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
    TSMC
end
if (TSMC>=0)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
    TSMC
end
if (TSMC>=0)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
    TSMC
end
if (TSMC>=0)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
    TSMC
end

if (TSMC>=0)
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx
    guessgamma=solvegamma
    ret
    TSMC
end
#############################################################
############################################

TSMC

opt=NLopt.Opt(:LN_NELDERMEAD,dg)
toluser=1e-6
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
gammav0=randn(dg)*1000
    #gammav0[:]=gamma1
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n

solvegamma=minx
guessgamma=solvegamma
ret
TSMC

opt=NLopt.Opt(:LN_SBPLX,dg)
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



opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=1e-12
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
gammav0=randn(dg)*1000
    #gammav0[:]=gamma1
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC

(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC

(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC

(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC

(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC

(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC

Results1=DataFrame([theta0 TSMC])
names!(Results1,Symbol.(["theta0","TSGMMcueMC"]))
Results1gamma=DataFrame(hcat(solvegamma,solvegamma))

CSV.write(dirresults*"/IncomeUncertaintymean_1.5million__results_TS_cuda_$theta0.vpilot.csv",Results1)
CSV.write(dirresults*"/IncomeUncertaintymean_1.5million__results_gamma_cuda_$theta0.vpilot.csv",Results1gamma)

Results1

print(Results1)
