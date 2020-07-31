#Version Julia "1.1.0"
#Author: Victor H. Aguiar & Nail Kashaev
#email: vhaguiar@gmail.com

count = 0
#Set the number of processors: Change it to the max the computer allows
nprocs=30
using Distributed
addprocs(nprocs)
@everywhere Distributed
@everywhere using Random
# Set a random seed in each processor

@everywhere using NLopt
@everywhere using DataFrames
@everywhere using MathProgBase
using CSV
@everywhere using RCall
@everywhere using LinearAlgebra
## set directory
machine="niagara"
if machine=="niagara"
  rootdir="/gpfs/fs1/home/v/vaguiar/vaguiar/ReplicationAK/SecondApp"
  dir=rootdir*"/data"
  dirresults="/gpfs/fs0/scratch/v/vaguiar/vaguiar/results"
end
#results file
if machine=="nailmachine"
  ##the following line is eliminated in case the installation of Rcall happened correctly and there is only 1 installation of R
  @everywhere ENV["R_HOME"]="C:\\Users\\Nkashaev\\Documents\\R\\R-3.4.4"
  rootdir="C:/Users/Nkashaev/Dropbox/ReplicationAK/SecondApp"
  dir=rootdir*"/data"
  dirresults=rootdir*"/results"
end

# data size
@everywhere  n=154

## time length
@everywhere const T=50
#@everywhere const T=4
## number of goods
@everywhere const K=3
# repetitions foreffective rejection sampling
## because the simulations are done using parallel Montecarlo we have nsimps*nprocs draws.
# set burn, if needed
burnrate=0
nsimsp=30
@everywhere const repn=($burnrate,$nsimsp)
nsims=nsimsp*nprocs
## Define the constant number of proccesors
const nprocs0=nprocs


# read csv files prepared in R from the original data in Ahn et al.
# prices array
dum0=CSV.read(dir*"/rationalitydata3goods.csv")
# break the dataset into subdatasets by individual
splitdum0=groupby(dum0,:id)
@eval @everywhere splitp=$splitdum0

#Initialize array of effective prices, for this application \rho= p
@everywhere  rho=zeros(n,T,K)
#Initialize array of consumption
@everywhere  cve=zeros(n,T,K)

#Fill the arrays
# Columns 10-12 correspond to prices
# Columns 4:6 correspond to consumption bundles
@everywhere for i=1:n
    dum0=convert(Array,splitp[i])
    rho[i,:,:]=dum0[1:T,10:12]
    cve[i,:,:]=dum0[1:T,4:6]
end


################################################################
##Output for simulations parameters
# Number of centering moments w^p\in R^(T*K)
@everywhere dg=T*K
@everywhere nsims=1
@everywhere ndelta=1
@everywhere solv=zeros(nsims,ndelta)
@everywhere solvgamma=zeros(nsims,ndelta,dg)
@everywhere solvwgamma=zeros(nsims,ndelta,dg)
@everywhere solvw=zeros(nsims,ndelta)
AGs=zeros(nsims,ndelta,2)
results=hcat(solv[1,:],solvw[1,:],solvgamma[1,:,:],solvwgamma[1,:,:])


###########################################################
### Main functions
###########################################################

## Centering Moments Function
##Fast myfun
#d.- discount factor, here it is set to 1.
#gamma.- passing zero matrix of the (dg x 1) size
#eta.- passes the simulated data, here it is equal to the simulated p^* (true price) satisfying GARP given quantities and budget constraints
#U.- When active it passes utility numbers from the Afriat inequalities, here it is not active.
#W.- passes zero array to be filled with the simulated error, here it is filled by the moments per each individual observation
#gvec.- passes zero vector to be filled the average of moments
#dummf.- auxiliary zero vector for the algorith; here not active
#cve.- consumption array
#rho.- effective price array

## AK_myfunc_gabaix.jl tests for E[w^p]=0
include(rootdir*"/secondappfunctions/myfun_pm.jl")

## New Guess Functions: Constraint to satisfy pw=0 as.
@everywhere m=zeros(n,T)
@everywhere mrep=zeros(n,T,K)
@everywhere msim=zeros(n,T)
@everywhere cvesim=zeros(n,T,K)
@everywhere YMat=zeros(n,T,K)
@everywhere XMat=zeros(n,T,K)
@everywhere for t=1:T
  m[:,t]=sum((cve[:,t,:]).*(rho[:,t,:]),dims=2)
end
@everywhere for k=1:K
  mrep[:,:,k]=m[:,:]
end
@everywhere mmax=maximum(m)
@everywhere mmin=minimum(m)
@everywhere ptest=ones(T,K)
@everywhere wtest=ones(T,K)

#Guessfun: gives the initial draw of the Montecarlo step, must be a simulations consistent with the null.

## Here it invokes the revealedPrefsmod function simGarpQuantWealth, that will generate a draw of p^* that satisfies GARP and in on the budget line.
## The function allows to set an afriatpar that corresponds to the cost efficiency index. We set it to 1.
#maxit is the max. number of iterations allowed by the sampler before it restarts.
#R has to get a random seed.

include(rootdir*"/secondappfunctions/guessfun_price.jl")


## Here it invokes the revealedPrefsmod function simGarpQuantWealth, that will generate a draw of p^* that satisfies GARP and in on the budget line.
## The function allows to set an afriatpar that corresponds to the cost efficiency index. We set it to 1.
#maxit is the max. number of iterations allowed by the sampler before it restarts.
#R has to get a random seed.
#Do not pay attention to the name of the files cvesim since it does not matter, in this case it is filled by prices
#include(rootdir*"/AK_guessfunc_quantityexperimental.jl")



###############################################################
## New Fast jump
## This function will draw new candidates for the Montecarlo, in this case this is the same as the guessfun.
## The reason is that in this case, we can generate exactly data under the null of GARP plus being on the budget.

## For prices
include(rootdir*"/secondappfunctions/jumpfun_price.jl")



## The Montecarlo step: It gives the integrated moments h
## This code follows Schennach's code in Gauss in the Supplement in ECMA for ELVIS.
##moments for generating the variance matrix: It generates the h and the g moments without averaging for building Omega.
@everywhere function gavraw(;d=d::Float64,gamma=gamma::Float64,myfun=myfun::Function,guessfun=guessfun::Function,jumpfun=jumpfun::Function,repn=repn,a=a::Array{Float64,2},gvec=gvec::Array{Float64,2},tryun=tryun::Array{Float64,2},trydens=trydens::Array64,eta=eta::Float64,U=U::Float64,W=W::Float64,dummf=dummf::Array{Float64,2},cve=cve::Float64,rho=rho::Float64)
  eta=guessfun(d=d,gamma=gamma,cve=cve,rho=rho)
  r=-repn[1]+1
  while r<=repn[2]
      tryun=jumpfun(d=d,gamma=gamma,cve=cve,rho=rho)
      logtrydens=myfun(d=d,gamma=gamma,eta=tryun,U=U,W=W,gvec=gvec,dummf=dummf,cve=cve,rho=rho)*gamma-myfun(d=d,gamma=gamma,eta=eta,U=U,W=W,gvec=gvec,dummf=dummf,cve=cve,rho=rho)*gamma
      dum=log.(rand(n)).<logtrydens
      @inbounds eta[dum,:]=tryun[dum,:]
      if r>0
        a=a+myfun(d=d,gamma=gamma,eta=eta,U=U,W=W,gvec=gvec,dummf=dummf,cve=cve,rho=rho)
      end
      r=r+1
    end
    a/repn[2]
end


## This function wraps up gavraw for parallelization, here it is just a wrapper.
@everywhere function dgavf(;d=d::Float64,gamma=gamma::Float64,myfun=myfun::Function,guessfun=guessfun::Function,jumpfun=jumpfun::Function,repn=repn,a=a::Array{Float64,2},gvec=gvec::Array{Float64,2},tryun=tryun::Array{Float64,2},trydens=trydens::Array64,eta=eta::Float64,U=U::Float64,W=W::Float64,dummf=dummf::Array{Float64,2},cve=cve::Float64,rho=rho::Float64)
  dvec0= @sync @distributed (+) for i=1:nprocs0
    Random.seed!(3000*i+Int(ceil(maximum(gamma))))
    gavraw(d=d,gamma=gamma,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
  end
  dvec0/nprocs0
end

## Specify the system tolerance for the optimization step in NLopt, set to 1e-6, for speed 1e-3 seems to be doing the same
@everywhere toluser=1e-6



##########################################################################
##########################################################################
## Initialize memory matrices
cdums=zeros(n,K)
gvec=zeros(n,(dg))
dummf=zeros(n,T,T)
U=zeros(n,T)
eta=zeros(n,dg)
W=zeros(n,T,K)
dummw0=zeros(n,1,K)
a=zeros(n,dg)
tryun=zeros(n,(dg))
eta=zeros(n,(dg))
trydens=zeros(n)


######################################################################
## GMM
i=1
ind=1
d0=1.0
#########################################################################
## Weighted Objective


function obj2(gamma0::Vector, grad::Vector)
  if length(grad) > 0
  end
  eta=guessfun(d=d0,gamma=gamma0,cve=cve,rho=rho)
  ###Solve
  dum=dgavf(d=d0,gamma=gamma0,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
  inddummy0=zeros(dg).<ones(dg)
  dvecdum=sum(dum,dims=1)[:,inddummy0]
  vardum=zeros(dg,dg)
  for j=1:n
    vardum=vardum+dum[j,inddummy0]*dum[j,inddummy0]'
  end
  ##Computing Ω
  vardum2=vardum/n-(dvecdum/n)'*(dvecdum/n)
  ##Find the inverse
  (Lambda,QM)=eigen(vardum2)
  inddummy=Lambda.>0.00001
  An=QM[:,inddummy]

  dvecdum2=An'*(dvecdum/n)'
  vardum3=An'*vardum2*An
  Omega2=inv(vardum3)
  Qn2=1/2*dvecdum2'*Omega2*dvecdum2
  return Qn2[1]
end

Random.seed!(3000)
gammav0=randn(dg)

# Set a random seed in each processor
@distributed for replicate_idx=1:nprocs
  Random.seed!(3000*replicate_idx)
end


opt=NLopt.Opt(:LN_BOBYQA,dg)
NLopt.lower_bounds!(opt,vcat(ones(dg).*-Inf))
NLopt.upper_bounds!(opt,vcat(ones(dg).*Inf))
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,obj2)
(minf,minx,ret) = NLopt.optimize!(opt, gammav0)
solvw[ind,i]=minf*2*n
solvwgamma[ind,i,:]=minx



results=hcat(solvw[1,:],solvwgamma[1,:,:])

#########################################################################
## Export
DFsolv=convert(DataFrame,results)
CSV.write(dirresults*"/2App_pm_reps_900.csv",DFsolv)
##########################################################################
##########################################################################
