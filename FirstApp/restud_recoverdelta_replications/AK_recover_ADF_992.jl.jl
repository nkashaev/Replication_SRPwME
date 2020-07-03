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
theta0=0.975
avgdelta=0.992


################################################################################
## Setting-up directory

computer="teslamachine"
appname="restud_recoverdelta2"
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
const T=4
const dg=5


###############################################################################
## Data
## Price data from Adams et al.

##seed
dataapp="singles"
## sample size
#singles
if dataapp=="singles"
     const n=185
end

if dataapp=="couples"
     const n=2004
end
## time length of the original data
T0=4
## number of goods
const K=17
# repetitions for the simulation
## because the simulations are done using parallel Montecarlo we have 100*nprocs draws.
const repn=(0,1000000)




###############################################################################
## Data
## Price data from Adams et al.

if dataapp=="singles"
    dir=rootdir*"/singles"
    dirresults=rootdir*"/singles/results"

    dum0=CSV.read(dir*"/p.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    dum0=reshape(dum0,n,T0,K)
    @eval  const p=$dum0
    ## Consumption data from Adams et al.
    dum0=CSV.read(dir*"/cve.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    ##original scale in the dataset
    dum0=reshape(dum0,n,T0,K)
    @eval   cve=$dum0

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
    dum0=reshape(dum0,n,T0,K)
    @eval const p=$dum0
    # consumption array
    dum0=CSV.read(dir*"/cvecouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    #dum0=reshape(dum0,n,T,K)./1e5
    dum0=reshape(dum0,n,T0,K)./1e5
    @eval const cve=$dum0

    # interest rate array
    dum0=CSV.read(dir*"/rvcouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    @eval const rv=$dum0.+1


end;



###############################################################################
## Data Cleaning, Counterfactual prices
 rho=zeros(n,T,K)

## Discounted prices
for i=1:n
  for t=1:(T0)
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
  end
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
#include(rootdir*"/cpufunctions/myfun_counter.jl")
include(rootdir*"/cpufunctions/myfun_recoverdelta.jl")
## chain generation with CUDA
chainM=zeros(n,dg,repn[2])
include(rootdir*"/cudafunctions/cuda_chainfun.jl")
## optimization with CUDA
numblocks = ceil(Int, n/100)
const nfast=20000
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
Random.seed!(123)
indfast=rand(1:repn[2],nfast)
indfast[1]=1
chainMcu[:,:,:]=cu(chainM[:,:,indfast])


###############################################################################
###############################################################################
Random.seed!(123)
res = bboptimize(objMCcu2c; SearchRange = (-10e300,10e300), NumDimensions = dg,MaxTime = 100.0, TraceMode=:silent)


minr=best_fitness(res)
TSMC=2*minr*n
TSMC
guessgamma=best_candidate(res)

#guessgamma=[0; 4.67E+300; 0; 0; -7.43E+300]
###############################################################################
###############################################################################


opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=1e-6
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)

#try 1
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret

#try 2
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

#try 4
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret

#try 5
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret


#try 6
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret


#try 7
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret

#try 8
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx
guessgamma=solvegamma
ret

#############################################################
############################################

TSMC





Results1=DataFrame([repn[2] TSMC])
names!(Results1,Symbol.(["theta0","TSGMMcueMC"]))
#names!(Results1,Symbol.(["theta0","TSGMMnovar","TSGMMcue","TSGMMcueMC"]))
Results1gamma=DataFrame(hcat(solvegamma,solvegamma))
#CSV.write(dirresults*"/chainM50K.csv",DataFrame(reshape(chainM,n*T,repn[2])))
#CSV.write()
CSV.write(dirresults*"/results_ADF_cuda_$avgdelta.sims_$repn[2].csv",Results1)
CSV.write(dirresults*"/results_gamma_ADF_$avgdelta.sims_$repn[2].csv",Results1gamma)

Results1

print(Results1)
