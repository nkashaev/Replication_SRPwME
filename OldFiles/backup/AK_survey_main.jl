#Version Julia "0.5.0"
#Author: Victor H. Aguiar
#email: vhaguiar@gmail.com

count = 0
#Set the number of processors: Change it to the max the computer allows
nprocs=4
addprocs(nprocs)
# Set a random seed
@eval @everywhere vern=6
@everywhere srand(vern*tic())
@everywhere using NLopt
@everywhere using DataFrames
@everywhere using MathProgBase
@everywhere using Clp

##the following line is eliminated in case the installation of Rcall happened correctly and there is only 1 installation of R
# @everywhere ENV["R_HOME"]="C:\\Users\\vaguiar\\Documents\\R\\R-3.3.1"
# @everywhere using RCall
## set directory
#Dusky cluster dir:
#rootdir="/home/vaguiar/AKsims"
## Office computer
rootdir="D:/Dropbox/AKsource/AKsimuncertaintyapplication"


dir=rootdir*"/Application_Single"


#results file
dirresults=rootdir*"/Application_Single/results"

# data size
@everywhere  n=185

## time length
## for this demo I am cutting the lenght of the data by 5 so it shows the algorithm in a reasonable time
#@everywhere const T=50
@everywhere const T=4
## number of goods
@everywhere const K=17
# repetitions for the simulation
## because the simulations are done using parallel Montecarlo we have 100*nprocs draws.
# set burn
burnrate=1
nsimsp=10
@everywhere const repn=($burnrate,$nsimsp)
## Define the constant number of proccesors
const nprocs0=nprocs


# read csv files prepared in R from the original data in Ahn et al.
# prices array
# read csv files prepared in R from the original consumption panel AER 2014, time discount and household decision making.
# prices array
# read csv files prepared in R from the original consumption panel AER 2014, time discount and household decision making.
# prices array
dum0=readtable(dir*"/p.csv")
dum0=convert(Array,dum0[:,:])
dum0=reshape(dum0,n,T,K)
@eval @everywhere  p=$dum0
# # consumption array
dum0=readtable(dir*"/cve.csv")
dum0=convert(Array,dum0[:,:])
dum0=reshape(dum0,n,T,K)
@eval @everywhere  cve=$dum0

# # interest rate array
dum0=readtable(dir*"/rv.csv")
dum0=convert(Array,dum0[:,:])
@eval @everywhere const rv=$dum0+1



@everywhere  rho=zeros(n,T,K)


@everywhere for i=1:n
  for t=1:T
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
  end
end

@everywhere function rand_uniform(n,a, b)
    a + rand(n)*(b - a)
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

## myfunc_gabaix.jl tests for E[w^p]=0 or E[p^*]=E[p].
#@everywhere include($rootdir*"/AK_myfunc_gabaix.jl")

## AK_myfunc_tremblinghand.jl tests for E[w^c]=0 or E[c^*]=E[c].
#include(rootdir*"/AK_myfunc_tremblinghand.jl")
include(rootdir*"/functions/myfun_surveyuncertain.jl")

##New Guess Functions: Constraint to satisfy pw=0 as.
ind=1
q=cve[ind,:,:]'
sol = linprog([-1,0],[2 1],'<',1.5, ClpSolver())
solsuccess=sol.status
@everywhere stepdum= .1
@everywhere deltat=collect(.1:stepdum:2)
soldet=zeros(size(deltat,1),n)
@everywhere function lccons(;delta=delta,p=p,q=q,A=A,b=b)
  for i=1:size(q,2)
    tmp1=-eye(size(q,2))
    tmp1[:,i]=ones(size(q,2),1)
    tmp1[i,:]=zeros(1,size(q,2))
    tmp2=zeros(size(q,2),1)
    for j=1:size(q,2)
      tmp2[j,1]=(delta^(-j+1)*(p[:,j]'*(q[:,i]-q[:,j])))[1]
    end
    #if i==1
    A[i,:,:]=tmp1
    b[i,:,:]=tmp2
    #print(A)
    #end
    #if i>1
    #  A=[A;tmp1]
    #  b=[b;tmp2]
    #end
  end
  (A,b)
end

## The Jump function: it gives a non retricted jump
@everywhere function guessfun(;d=d::Float64,gamma=gamma::Float64,cve=cve::Float64,rho=rho::Float64)
  drawsim=hcat(zeros(n),zeros(n,T),zeros(n,T),zeros(n,K*T))

  randelta=rand(n)*(1-d)+d
  #randlambda=rand(n,T).*(1.5-.5)+.5
  randlambda=randexp(n,T)
  randlambda[:,1]=ones(n)
  #cvesim=randexp(n,T,K)
  randw=randexp(n,T)/10
  cvesim=log.(1./rand(n,T,K))
  rhosim=zeros(n,T,K)
  for t in 1:T
    dumgen=sum(cvesim[:,t,:],2)[1]
    for k in 1:K
      cvesim[:,t,k]=cvesim[:,t,k]./dumgen./rho[:,t,k].*randw[:,t]
      rhosim[:,t,k]=rho[:,t,k].*randlambda[:,t]
    end
  end
  sol = linprog([-1,0],[2 1],'<',1.5, ClpSolver())
  solsuccess=sol.status
  soldet=zeros(n)
  solsol=zeros(n,T)
  #Initialization
  rate=0
  ## simulations
  while rate < 1
    for ind=1:n
      if soldet[ind]==0

        A0=zeros(T,T,T)
        b0=zeros(T,T,1)


        (Ar,br)=lccons(delta=randelta[ind],p=rhosim[ind,:,:]',q=cvesim[ind,:,:]',A=A0,b=b0)

        Ar=reshape(Ar,T*T,T)
        br=reshape(br,T*T,1)
        ind0=zeros(size(Ar,1))

          for i=1:size(Ar,1)
            if Ar[i,:]==zeros(size(Ar,2))
              ind0[i]=false
            else
              ind0[i]=true
            end
          end

          ind0=ind0.==ones(size(Ar,1))
          Ar=Ar[ind0,:]
          br=br[ind0,:]

          c=zeros(T)

          lb=ones(T)

          ub=Inf*ones(T)
          ##This works
          sol = linprog(c,Ar,'<',br[:,1],lb,ub,ClpSolver())
          soldet[ind]=sol.status==solsuccess

          if soldet[ind]==0
            cvesim[ind,:,:]=randexp(T,K)
          end
          if soldet[ind]==1
            solsol[ind,:]=sol.sol
          end
      end
    rate=sum(soldet)/n
    end
  end
  drawsim[:,1]=randelta
  drawsim[:,2:(T+1)]=randlambda
  drawsim[:,(T+2):(2*T+1)]=solsol
  drawsim[:,(2*T+2):end]=reshape(cvesim,n,T*K)
  drawsim
end

## The Jump functions
## The Jump function: it gives a non retricted jump
@everywhere function jumpfun(;d=d::Float64,gamma=gamma::Float64,cve=cve::Float64,rho=rho::Float64)
  drawsim=hcat(zeros(n),zeros(n,T),zeros(n,T),zeros(n,K*T))

  randelta=rand(n)*(1-d)+d
  #randlambda=rand(n,T).*(1.5-.5)+.5
  randlambda=randexp(n,T)
  randlambda[:,1]=ones(n)
  #cvesim=randexp(n,T,K)
  randw=randexp(n,T)/10
  cvesim=log.(1./rand(n,T,K))
  rhosim=zeros(n,T,K)
  for t in 1:T
    dumgen=sum(cvesim[:,t,:],2)[1]
    for k in 1:K
      cvesim[:,t,k]=cvesim[:,t,k]./dumgen./rho[:,t,k].*randw[:,t]
      rhosim[:,t,k]=rho[:,t,k].*randlambda[:,t]
    end
  end
  sol = linprog([-1,0],[2 1],'<',1.5, ClpSolver())
  solsuccess=sol.status
  soldet=zeros(n)
  solsol=zeros(n,T)
  #Initialization
  rate=0
  ## simulations
  while rate < 1
    for ind=1:n
      if soldet[ind]==0

        A0=zeros(T,T,T)
        b0=zeros(T,T,1)


        (Ar,br)=lccons(delta=randelta[ind],p=rhosim[ind,:,:]',q=cvesim[ind,:,:]',A=A0,b=b0)

        Ar=reshape(Ar,T*T,T)
        br=reshape(br,T*T,1)
        ind0=zeros(size(Ar,1))

          for i=1:size(Ar,1)
            if Ar[i,:]==zeros(size(Ar,2))
              ind0[i]=false
            else
              ind0[i]=true
            end
          end

          ind0=ind0.==ones(size(Ar,1))
          Ar=Ar[ind0,:]
          br=br[ind0,:]

          c=zeros(T)

          lb=ones(T)

          ub=Inf*ones(T)
          ##This works
          sol = linprog(c,Ar,'<',br[:,1],lb,ub,ClpSolver())
          soldet[ind]=sol.status==solsuccess

          if soldet[ind]==0
            cvesim[ind,:,:]=randexp(T,K)
          end
          if soldet[ind]==1
            solsol[ind,:]=sol.sol
          end
      end
    rate=sum(soldet)/n
    end
  end
  drawsim[:,1]=randelta
  drawsim[:,2:(T+1)]=randlambda
  drawsim[:,(T+2):(2*T+1)]=solsol
  drawsim[:,(2*T+2):end]=reshape(cvesim,n,T*K)
  drawsim
end


## The Montecarlo step: It gives the integrated moments h
## This code follows Schennach's code in Gauss in the Supplement in ECMA for ELVIS.
@everywhere function gavg(;d=d::Float64,gamma=gamma::Float64,myfun=myfun::Function,guessfun=guessfun::Function,jumpfun=jumpfun::Function,repn=repn,a=a::Array{Float64,2},gvec=gvec::Array{Float64,2},tryun=tryun::Array{Float64,2},trydens=trydens::Array64,eta=eta::Float64,U=U::Float64,W=W::Float64,dummf=dummf::Array{Float64,2},cve=cve::Float64,rho=rho::Float64)
  eta=guessfun(d=d,gamma=gamma,cve=cve,rho=rho)
  r=-repn[1]+1
  while r<=repn[2]
      tryun=jumpfun(d=d,gamma=gamma,cve=cve,rho=rho)
      logtrydens=myfun(d=d,gamma=gamma,eta=tryun,U=U,W=W,gvec=gvec,dummf=dummf,cve=cve,rho=rho)*gamma-myfun(d=d,gamma=gamma,eta=eta,U=U,W=W,gvec=gvec,dummf=dummf,cve=cve,rho=rho)*gamma
      dum=log(rand(n)).<logtrydens
      @inbounds eta[dum,:]=tryun[dum,:]
      if r>0
        a=a+myfun(d=d,gamma=gamma,eta=eta,U=U,W=W,gvec=gvec,dummf=dummf,cve=cve,rho=rho)
      end
      r=r+1
    end
    sum(a,1)/repn[2]
end


##moments for generating the variance matrix: It generates the h and the g moments without averaging for building Omega.
@everywhere function gavraw(;d=d::Float64,gamma=gamma::Float64,myfun=myfun::Function,guessfun=guessfun::Function,jumpfun=jumpfun::Function,repn=repn,a=a::Array{Float64,2},gvec=gvec::Array{Float64,2},tryun=tryun::Array{Float64,2},trydens=trydens::Array64,eta=eta::Float64,U=U::Float64,W=W::Float64,dummf=dummf::Array{Float64,2},cve=cve::Float64,rho=rho::Float64)
  eta=guessfun(d=d,gamma=gamma,cve=cve,rho=rho)
  r=-repn[1]+1
  while r<=repn[2]
      tryun=jumpfun(d=d,gamma=gamma,cve=cve,rho=rho)
      logtrydens=myfun(d=d,gamma=gamma,eta=tryun,U=U,W=W,gvec=gvec,dummf=dummf,cve=cve,rho=rho)*gamma-myfun(d=d,gamma=gamma,eta=eta,U=U,W=W,gvec=gvec,dummf=dummf,cve=cve,rho=rho)*gamma
      dum=log(rand(n)).<logtrydens
      @inbounds eta[dum,:]=tryun[dum,:]
      if r>0
        a=a+myfun(d=d,gamma=gamma,eta=eta,U=U,W=W,gvec=gvec,dummf=dummf,cve=cve,rho=rho)
      end
      r=r+1
    end
    a/repn[2]
end

## This function wraps up gavg for parallelization, here it is just a wrapper.
@everywhere function dvecf(;d=d::Float64,gamma=gamma::Float64,myfun=myfun::Function,guessfun=guessfun::Function,jumpfun=jumpfun::Function,repn=repn,a=a::Array{Float64,2},gvec=gvec::Array{Float64,2},tryun=tryun::Array{Float64,2},trydens=trydens::Array64,eta=eta::Float64,U=U::Float64,W=W::Float64,dummf=dummf::Array{Float64,2},cve=cve::Float64,rho=rho::Float64)
  dvec0= @sync @parallel (+) for i=1:nprocs0
    gavg(d=d,gamma=gamma,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
  end
  dvec0/nprocs0
end

## This function wraps up gavraw for parallelization, here it is just a wrapper.
@everywhere function dgavf(;d=d::Float64,gamma=gamma::Float64,myfun=myfun::Function,guessfun=guessfun::Function,jumpfun=jumpfun::Function,repn=repn,a=a::Array{Float64,2},gvec=gvec::Array{Float64,2},tryun=tryun::Array{Float64,2},trydens=trydens::Array64,eta=eta::Float64,U=U::Float64,W=W::Float64,dummf=dummf::Array{Float64,2},cve=cve::Float64,rho=rho::Float64)
  dvec0= @sync @parallel (+) for i=1:nprocs0
    gavraw(d=d,gamma=gamma,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
  end
  dvec0/nprocs0
end

## Specify the system tolerance for the optimization step in NLopt, set to 1e-6, for speed 1e-3 seems to be doing the same
@everywhere toluser=1e-3

#tic()
for ind=1:1
  #Set a line search over the lowerbounds of the discount factor
  # In this case this is innesential so set it up to anything.
  #deltat=[0.1 0.15 .2 .25 .3 .35 .4 .45 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95][ind]
  deltat=[.01][ind]
     n0=n



  ##########################################################################
  ##########################################################################
  ## First step GMM
  ## Initialize memory matrices
  cdums=zeros(n,K)
  dg=T*K
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

  ## Loop for values of δ
  i=1
  gammav0=randn(dg)
  while i<=length(deltat)
    d0=deltat[i]

    function obj2(gamma0::Vector, grad::Vector)
      if length(grad) > 0
      end
      eta=guessfun(d=d0,gamma=gamma0,cve=cve,rho=rho)
      dvec0=dvecf(d=d0,gamma=gamma0,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
      return sum((dvec0).^2)/(n0^2)
    end
    #Both LN_NELDERMEAD and LN_BOBYQA do the work, but the second is the fastest altenative derivative-free
    #opt=NLopt.Opt(:LN_NELDERMEAD,ltt2)
    opt=NLopt.Opt(:LN_BOBYQA,dg)
    NLopt.lower_bounds!(opt,vcat(ones(1:dg).*-Inf))
    NLopt.upper_bounds!(opt,vcat(ones(1:dg).*Inf))
    NLopt.xtol_rel!(opt,toluser)
    NLopt.min_objective!(opt,obj2)
    (minf,minx,ret) = NLopt.optimize!(opt, gammav0)
    solv[ind,i]=minf
    solvgamma[ind,i,:]=minx
    gammav0=randn(dg)
    i=i+1
  end


  #########################################################################
  ## Weighted Objective: Second Step GMM
  ## Loop for different values of δ
  i=1
  gammav0=randn(dg)

  while i<=length(deltat)
    d0=deltat[i]
    eta=guessfun(d=d0,gamma=gammav0,cve=cve,rho=rho)
    ## This step uses the first-step GMM gammav0
    dum=dgavf(d=d0,gamma=solvgamma[ind,i,:],myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
    inddummy0=zeros(dg).<ones(dg)
    dvecdum=sum(dum,1)[:,inddummy0]
    vardum=zeros(dg,dg)
    for j=1:n0
      vardum=vardum+dum[j,inddummy0]*dum[j,inddummy0]'
    end
    ## Computation of Ω
    vardum2=vardum/n0-(dvecdum/n0)'*(dvecdum/n0)
    ##Find the generalized inverse AG2017
    (Lambda,QM)=eig(vardum2)
    ## numerical zero
    inddummy=Lambda.>0.00001

    function obj2(gamma0::Vector, grad::Vector)
      if length(grad) > 0
      end
      eta=guessfun(d=d0,gamma=gamma0,cve=cve,rho=rho)
      ###Solve
      dum=dgavf(d=d0,gamma=gamma0,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
      inddummy0=zeros(dg).<ones(dg)
      dvecdum=sum(dum,1)[:,inddummy0]
      vardum=zeros(dg,dg)
      for j=1:n0
        vardum=vardum+dum[j,inddummy0]*dum[j,inddummy0]'
      end
      ##Computing Ω
      vardum2=vardum/n0-(dvecdum/n0)'*(dvecdum/n0)
      ##Find the inverse
      (Lambda,QM)=eig(vardum2)
      An=QM[:,inddummy]
      dvecdum2=An'*(dvecdum/n0)'
      vardum3=An'*vardum2*An
      Omega2=inv(vardum3)
      Qn2=1/2*dvecdum2'*Omega2*dvecdum2
      return Qn2[1]
    end

  #opt=NLopt.Opt(:LN_NELDERMEAD,ltt2+T)
    opt=NLopt.Opt(:LN_BOBYQA,dg)
    NLopt.lower_bounds!(opt,vcat(ones(1:dg).*-Inf))
    NLopt.upper_bounds!(opt,vcat(ones(1:dg).*Inf))
    NLopt.xtol_rel!(opt,toluser)
    NLopt.min_objective!(opt,obj2)
    (minf,minx,ret) = NLopt.optimize!(opt, gammav0)
    solvw[ind,i]=minf*2*n0
    solvwgamma[ind,i,:]=minx
    gammav0=randn(dg)
    i=i+1
  end


  results=hcat(solv[1,:],solvw[1,:],solvgamma[1,:,:],solvwgamma[1,:,:])
end
#toc()
#AGsfinal=hcat(deltat,AGs[:,1],AGs[:,2])
print(solv[1,:])

print(solvw[1,:])

results


###END simulation
#########################################################################
## Export
DFsolv=convert(DataFrame,results)
writetable(dirresults*"/results_experimentalgabaix900.csv",DFsolv)
#DFsolvw=convert(DataFrame,solvw)
#writetable(dirresults*"/solvwbetaeq1_err5.csv",DFsolvw)
##########################################################################
##########################################################################
