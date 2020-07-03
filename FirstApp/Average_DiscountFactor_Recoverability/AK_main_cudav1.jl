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
theta0=1

################################################################################
## Setting-up directory
#rootdir="D:/Dropbox/AKsource/AKEDapp"
computer="office"
if computer=="laptop"
    rootdir="D:/Dropbox/AKsource/AKEDapp"
end
if computer=="office"
    rootdir="D:/Dropbox/Dropbox/AKsource/AKEDapp"
end


################################################################################
##
# data size
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
## time length
@everywhere const T=4
## number of goods
@everywhere const K=17
# repetitions for the simulation
## because the simulations are done using parallel Montecarlo we have 100*nprocs draws.
@everywhere const repn=(0,50000)

chainM=zeros(n,T,repn[2])
## number of proccesors
const nprocs0=nprocsdum+1

###########################################
@everywhere const dg=T


###############################################################################
## Data
## Price data from Adams et al.

if dataapp=="singles"
    dir=rootdir*"/singles"
    dirresults=rootdir*"/power/results"

    dum0=CSV.read(dir*"/p.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    dum0=reshape(dum0,n,T,K)
    @eval @everywhere  const p=$dum0
    ## Consumption data from Adams et al.
    dum0=CSV.read(dir*"/cve.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    ##original scale in the dataset
    dum0=reshape(dum0,n,T,K)
    @eval @everywhere  const cve=$dum0
    #sum(ones(n,T,K).*cve,dims=3)

    ## Interest data from Adams et al.
    dum0=CSV.read(dir*"/rv.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    ## This step is done following the replication code in Adams et al.
    @eval @everywhere const rv=$dum0.+1


end;

if dataapp=="couples"
    dir=rootdir*"/couples"
    dirresults=rootdir*"/power/results"
    dum0=CSV.read(dir*"/pcouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    dum0=reshape(dum0,n,T,K)
    @eval @everywhere const p=$dum0
    # consumption array
    dum0=CSV.read(dir*"/cvecouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    #dum0=reshape(dum0,n,T,K)./1e5
    dum0=reshape(dum0,n,T,K)./1e5
    @eval @everywhere const cve=$dum0

    # interest rate array
    dum0=CSV.read(dir*"/rvcouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
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
numblocks = ceil(Int, n/167)
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


## warmstart
include(rootdir*"/restud_apppower/cpufunctions/warm_start_searchdelta.jl")
print("warm start ready!")


Random.seed!(123)
gchaincu!(theta0,gammav0,cve,rho,chainM)
print("chain ready!")

###################################################################################
############################################################################################
## Loading old results for optimization
preload=0
if preload==1
    using HDF5
    # h5write("C:/Users/vaguiar/Documents/AKSourcebinaries/chaincouples.h5", "cvesim", cvesim)
    # h5write("C:/Users/vaguiar/Documents/AKSourcebinaries/chaincouples.h5", "vsim", vsim)
    # h5write("C:/Users/vaguiar/Documents/AKSourcebinaries/chaincouples.h5", "Delta", Delta)

    cvesim=h5read("C:/Users/vaguiar/Documents/AKSourcebinaries/chaincouples.h5", "cvesim")
    vsim=h5read("C:/Users/vaguiar/Documents/AKSourcebinaries/chaincouples.h5", "vsim")
    Delta=h5read("C:/Users/vaguiar/Documents/AKSourcebinaries/chaincouples.h5", "Delta")
    W=cve-cvesim

    # chainMbin=collect(chainM)
    # h5write("C:/Users/vaguiar/Documents/AKSourcebinaries/chaincouplesd1.h5", "chainM1", chainM)
    # chainMbin=nothing
    # GC.gc()

    chainM = h5read("C:/Users/vaguiar/Documents/AKSourcebinaries/chaincouplesd1.h5", "chainM1")
    minimum(Delta)
    maximum(Delta)
    norm(chainM)


end

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
res = bboptimize(objMCcu2; SearchRange = (-10e5,10e5), NumDimensions = 4,MaxTime = 100.0, TraceMode=:silent)
#res = bboptimize(objMCcu2; SearchRange = (-10e5,10e5), NumDimensions = 4,MaxTime = 100.0,Method=:probabilistic_descent)


minr=best_fitness(res)
TSMC=2*minr*n
TSMC
guessgamma=best_candidate(res)


###############################################################################
###############################################################################
## objMCcu(ones(T)*0,ones(T)*0)
opt=NLopt.Opt(:LN_NELDERMEAD,dg)
toluser=1e-6
NLopt.lower_bounds!(opt,ones(dg).*(1e10)*(-1))
NLopt.upper_bounds!(opt,ones(dg).*1e10)
#NLopt.lower_bounds!(opt,ones(dg).*-Inf)
#NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
gammav0=randn(dg)*1000
#gammav0[:]=gamma1
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
TSMC=2*minf*n
TSMC
solvegamma=minx



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
