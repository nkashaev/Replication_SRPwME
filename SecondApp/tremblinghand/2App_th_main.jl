# Sample size.
@everywhere  const n=154
# Number of time periods.
@everywhere const T=50
# Number of goods.
@everywhere const K=3
# Number of simulations per processor and number of burned elements of rejection sampling.
@everywhere const repn=($burnrate,$nsimsp)
nsims=nsimsp*nprocs
# Define the constant number of proccesors
const nprocs0=nprocs
################################################################################
## Data
# Read csv files prepared in R from the original data in Ahn et al. (2014).
dum0=CSV.read(dir*"/rationalitydata3goods.csv")
# Break the dataset into individual datasets.
splitdum0=groupby(dum0,:id)
@eval @everywhere splitp=$splitdum0
#Initialize array of effective prices. For this application \rho= p.
@everywhere  const rho=zeros(n,T,K).*1.0
#Initialize array of consumption.
@everywhere  const cve=zeros(n,T,K).*1.0
#Fill the arrays
# Columns 10-12 correspond to prices.
# Columns 4-6 correspond to consumption bundles.
@everywhere for i=1:n
    dum0=convert(Array,splitp[i])
    rho[i,:,:]=dum0[1:T,10:12]
    cve[i,:,:]=dum0[1:T,4:6]
end
################################################################################
## Output for simulations parameters.
# Number of moments: w^p\in R^(T*K).
@everywhere const dg=T*K
@everywhere nsims=1
@everywhere ndelta=1
@everywhere solv=zeros(nsims,ndelta)
@everywhere solvgamma=zeros(nsims,ndelta,dg)
@everywhere solvwgamma=zeros(nsims,ndelta,dg)
@everywhere solvw=zeros(nsims,ndelta)
AGs=zeros(nsims,ndelta,2)
results=hcat(solv[1,:],solvw[1,:],solvgamma[1,:,:],solvwgamma[1,:,:])

################################################################################
## Main functions
################################################################################
##Moments Function.
# This function is the moment function g(x,e).
include(rootdir*"/secondappfunctions/myfun_th.jl")

#New Guess Functions: Constraint to satisfy pw=0 as.
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
#Fuess function. Generates the initial draw of the Montecarlo step.
include(rootdir*"/secondappfunctions/guessfun_quantity.jl")

## This function will draw new candidates for the Montecarlo. In this case this is the same as the guessfun.
include(rootdir*"/secondappfunctions/jumpfun_quantity.jl")

## The Montecarlo step: It gives the integrated moments h.
# This code follows Schennach's code in Gauss (Schennach, 2014).

# Moments for generating the variance matrix: It generates h and g moments without averaging for building Omega.
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

# These functions wrap up  gavraw for parallelization.

@everywhere function dgavf(;d=d::Float64,gamma=gamma::Float64,myfun=myfun::Function,guessfun=guessfun::Function,jumpfun=jumpfun::Function,repn=repn,a=a::Array{Float64,2},gvec=gvec::Array{Float64,2},tryun=tryun::Array{Float64,2},trydens=trydens::Array64,eta=eta::Float64,U=U::Float64,W=W::Float64,dummf=dummf::Array{Float64,2},cve=cve::Float64,rho=rho::Float64)
  dvec0= @sync @distributed (+) for i=1:nprocs0
    Random.seed!(3000*i+Int(ceil(maximum(gamma))))
    gavraw(d=d,gamma=gamma,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
  end
  dvec0/nprocs0
end
################################################################################
## Optimization
# Specify the system tolerance for the optimization step in NLopt
@everywhere toluser=1e-6
# Initialize memory matrices
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

# GMM
# Setting the initial parameters.  Needed for compilation of functions below.
i=1
ind=1
d0=1.0

# Weighted Objective
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
# Initial value of gamma
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
