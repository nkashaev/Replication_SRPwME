#Version Julia "0.5.0"
#using MathProgBase
#using Clp
#using Gadfly

count = 0
nprocs=19
#nprocs=3
addprocs(nprocs)
count = 0
nprocs=19
@eval @everywhere vern=6
@everywhere srand(vern*tic())
#using Iterators
#import Optim
@everywhere using NLopt
@everywhere using DataFrames
@everywhere using MathProgBase
#@everywhere using Clp
#@everywhere ENV["R_HOME"]="C:\\Users\\vaguiar\\Documents\\R\\R-3.3.1"
@everywhere using RCall
## set directory
#rootdir="/home/adam/Dropbox/AKsource/AKsimsandapplicationstatic"
#rootdir="/home/vaguiar/AKsims"
#file names
#C:\Users\vaguiar\Dropbox
#dir="/home/ap/Dropbox/exponentialdiscountingELVIS/ELVISjulia/Application_Single"
## Office computer
#rootdir="C:/Users/vaguiar/Dropbox/AKsource/AKexperimental"
rootdir="/home/vaguiar/AKsims/AKexperimental"

dir=rootdir*"/Application_Couples"

#results file
#dirresults="/home/ap/Dropbox/exponentialdiscountingELVIS/ELVISjulia/simulation/results"
#dirresults="C:/Users/vaguiar/Dropbox/exponentialdiscountingELVIS/ELVISjulia/simulation/results"
#dirresults="D:/Dropbox/exponentialdiscountingELVIS/ELVISjulia/simulation/results"
dirresults=rootdir*"/Application_Couples/results"
# data size
##seed
@everywhere srand(3000)
## sample size
#singles
#@everywhere  n=185
#couples
@everywhere  n=154

## time length
@everywhere const T=50
## number of goods
@everywhere const K=3
# repetitions for the simulation
## because the simulations are done using parallel Montecarlo we have 100*nprocs draws.
##Default: @everywhere const repn=(199,699)
#@everywhere const repn=(199,699)
#@everywhere const repn=(9,199)
@everywhere const repn=(1,29)
#@everywhere const repn=(299,9999)
## number of proccesors
const nprocs0=nprocs+1


# read csv files prepared in R from the original consumption panel AER 2014, time discount and household decision making.
# prices array
# read csv files prepared in R from the original consumption panel AER 2014, time discount and household decision making.
# prices array
dum0=readtable(dir*"/rationalitydata3goods.csv")
splitdum0=groupby(dum0,:id)
@eval @everywhere splitp=$splitdum0
# dum0=convert(Array,dum0[:,10:12])
# dum0=reshape(dum0,n,T,K)
# @eval @everywhere p=$dum0
# # consumption array
# dum0=readtable(dir*"/rationalitydata3goods.csv")
# dum0=convert(Array,dum0[:,4:6])
# dum0=reshape(dum0,n,T,K)
# @eval @everywhere cve=$dum0

# interest rate array
# dum0=readtable(dir*"/rvcouple.csv")
# dum0=convert(Array,dum0[:,:])
# @eval @everywhere rv=$dum0+1



@everywhere  rho=zeros(n,T,K)
@everywhere  cve=zeros(n,T,K)

#@everywhere const cvecenterbig=zeros(n,T,length(tt2))


@everywhere for i=1:n
  #for t=1:T
    #rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
    dum0=convert(Array,splitp[i])
    rho[i,:,:]=dum0[:,10:12]
    cve[i,:,:]=dum0[:,4:6]
  #end
end

@everywhere function rand_uniform(n,a, b)
    a + rand(n)*(b - a)
end


### Sanity check
@everywhere index=zeros(n).<ones(n)
@everywhere cve=cve[index,1:2,:]
@everywhere rho=rho[index,1:2,:]
@everywhere  n=sum(index*1)
@everywhere  const T=2
##select
# @everywhere rhoorg=rho
# @everywhere cveorg=cve
# @everywhere nsample=200
# @everywhere norg=2004
# @everywhere inddata=sample(1:norg,nsample)
# @everywhere n=nsample
# @everywhere rho=rhoorg[inddata,:,:]
# @everywhere cve=cveorg[inddata,:,:]
# @everywhere n=size(inddata)[1]
################################################################
##Output for simulations parameters
#@everywhere dg=T*T+T
#@everywhere dg=T*T+T*K
@everywhere dg=T*K
@everywhere nsims=1
@everywhere ndelta=1
@everywhere solv=zeros(nsims,ndelta)
@everywhere solvgamma=zeros(nsims,ndelta,dg)
@everywhere solvwgamma=zeros(nsims,ndelta,dg)
@everywhere solvw=zeros(nsims,ndelta)
AGs=zeros(nsims,ndelta,2)
results=hcat(solv[1,:],solvw[1,:],solvgamma[1,:,:],solvwgamma[1,:,:])

# @everywhere n=100
# @everywhere rho=zeros(n,T,K)
# @everywhere cve=zeros(n,T,K)
#
# @everywhere for i=1:n
#   for t=1:T
#      rho[i,t,:]=randexp(K)'*10
#   end
# end

#'
###Simulated CVE
##Beta


#rho2=vcat(rho,rho)
#rho=rho2
#cve=rho
#n=n*2
##Indicator
#@everywhere ind=1
###########################################################
### Main functions
###########################################################

## Moments Function
##Fast myfun
@everywhere function myfun(;d=d::Float64,gamma=gamma::Float64,eta=eta::Float64,U=U::Float64,W=W::Float64,gvec=gvec::Array{Float64,2},dummf=dummf::Float64,cve=cve::Float64,rho=rho::Float64)
    #eta[:,1:T]=eta[:,1:T]
    #U[:]=reshape(eta[:,(T+1):(T+T)],n,T)
    W[:]=cve-reshape(eta,n,T,K)

    # for t = 1:T
    #   for s=1:T
    #
    #     @inbounds dummf[:,t,s]=(eta[:,t].^(-1)).*(U[:,t]-U[:,s])-(sum((cve[:,t,:]-cve[:,s,:]+W[:,s,:]-W[:,t,:]).*(rho[:,t,:]),2) )
    #
    #   end
    # end
    #
    #   @inbounds gvec[:,1:T*T]=reshape(dummf,n,T*T).<zeros(n,T*T)


    #@simd for j=(T*T+1):(dg)

      #@inbounds gvec[:,j]=(eta[:,j-(T*T)]).*sum((W[:,j-(T*T),:]).*(rho[:,j-(T*T),:]),2)/1e30
      #@inbounds gvec[:,j]=sum((W[:,j-(T*T),:]).*(rho[:,j-(T*T),:]),2)/1e30
      #@inbounds gvec[:,j]=(eta[:,j-(T*T)]).*sum((W[:,j-(T*T),:]).*(rho[:,j-(T*T),:]),2)/1e30
      #@inbounds gvec[:,j]=sum((W[:,j-(T*T),:]).*(rho[:,j-(T*T),:]),2)/1e30
    #end
    gvec[:,:]=reshape(W,n,T*K)/1000

    gvec
end

##New Guess Functions: Constraint to satisfy pw=0 as.
@everywhere m=zeros(n,T)
@everywhere mrep=zeros(n,T,K)
@everywhere msim=zeros(n,T)
@everywhere cvesim=zeros(n,T,K)
@everywhere YMat=zeros(n,T,K)
@everywhere XMat=zeros(n,T,K)
@everywhere for t=1:T
  m[:,t]=sum((cve[:,t,:]).*(rho[:,t,:]),2)
end
@everywhere for k=1:K
  mrep[:,:,k]=m[:,:]
end
@everywhere mmax=maximum(m)
@everywhere mmin=minimum(m)
@everywhere ptest=ones(T,K)
@everywhere wtest=ones(T,K)



###############################################################
## New Fast jump
@everywhere function jumpfun(;d=d::Float64,gamma=gamma::Float64,cve=cve::Float64,rho=rho::Float64)
    #hcat(rand(n),randexp(n,T)/1000000,randexp(n,K*T)/1000000)
    #YMat[:,:,:]=log(ones(n,T,K)./rand(n,T,K))
    #XMat[:,:,:]=YMat./sum(YMat,3)
    #m[:,:]=rand(n,T).*(mmax-mmin)+mmin
    nobs=T
    ngoods=K
    afriatpar=1
    seed=rand(1:1000000000)
    maxit=1000
    R"set.seed($seed)"
    for indz=1:n
      ptest=rho[indz,:,:]
      wtest=mrep[indz,:,:]
      R"res2=revealedPrefsmod::simGarpPriceWealth(nobs=$nobs,ngoods=$ngoods,afriat.par=$afriatpar,maxit=$maxit,qmin=0,qmax=1,pmin=0,pmax=1,p=$ptest,w=$wtest)"
      R"res2x=res2$x"
      #R"res2p=res2$p"
      #for k=1:K
      @rget res2x
      ntest=size(res2x)[1]
      maxit2=1
      while (ntest<T | maxit2<=100000)
        seed=rand(1:1000000000)
        R"set.seed($seed)"
        R"res2=revealedPrefsmod::simGarpPriceWealth(nobs=$nobs,ngoods=$ngoods,afriat.par=$afriatpar,maxit=$maxit,qmin=0,qmax=1,pmin=0,pmax=1,p=$ptest,w=$wtest)"
        R"res2x=res2$x"
        maxit2=maxit2+1
        @rget res2x
        ntest=size(res2x)[1]
      end
      #@rget res2x
      cvesim[indz,:,:]=res2x
      #end
    end
    #hcat(rand(n,T).*(1-d)+d,randexp(n,T),randexp(n,K*T))
    #hcat(rand(n,T).*(1-d)+d,randexp(n,T),reshape(cvesim,n,T*K))
    hcat(reshape(cvesim,n,T*K))
end

@everywhere function guessfun(;d=d::Float64,gamma=gamma::Float64,cve=cve::Float64,rho=rho::Float64)
    #hcat(rand(n),randexp(n,T)/1000000,randexp(n,K*T)/1000000)
    #YMat[:,:,:]=log(ones(n,T,K)./rand(n,T,K))
    #XMat[:,:,:]=YMat./sum(YMat,3)
    #m[:,:]=rand(n,T).*(mmax-mmin)+mmin
    nobs=T
    ngoods=K
    afriatpar=1
    seed=rand(1:1000000000)
    maxit=1000
    R"set.seed($seed)"
    for indz=1:n
      ptest=rho[indz,:,:]
      wtest=mrep[indz,:,:]
      R"res2=revealedPrefsmod::simGarpPriceWealth(nobs=$nobs,ngoods=$ngoods,afriat.par=$afriatpar,maxit=$maxit,qmin=0,qmax=1,pmin=0,pmax=1,p=$ptest,w=$wtest)"
      R"res2x=res2$x"
      #R"res2p=res2$p"
      #for k=1:K
      @rget res2x
      ntest=size(res2x)[1]
      maxit2=1
      while (ntest<T | maxit2<=100000)
        seed=rand(1:1000000000)
        R"set.seed($seed)"
        R"res2=revealedPrefsmod::simGarpPriceWealth(nobs=$nobs,ngoods=$ngoods,afriat.par=$afriatpar,maxit=$maxit,qmin=0,qmax=1,pmin=0,pmax=1,p=$ptest,w=$wtest)"
        R"res2x=res2$x"
        maxit2=maxit2+1
        @rget res2x
        ntest=size(res2x)[1]
      end
      #@rget res2x
      cvesim[indz,:,:]=res2x
      #end
    end
    #hcat(rand(n,T).*(1-d)+d,randexp(n,T),randexp(n,K*T))
    #hcat(rand(n,T).*(1-d)+d,randexp(n,T),reshape(cvesim,n,T*K))
    hcat(reshape(cvesim,n,T*K))
end



## The Montecarlo step: It gives the integrated moments h
@everywhere function gavg(;d=d::Float64,gamma=gamma::Float64,myfun=myfun::Function,guessfun=guessfun::Function,jumpfun=jumpfun::Function,repn=repn,a=a::Array{Float64,2},gvec=gvec::Array{Float64,2},tryun=tryun::Array{Float64,2},trydens=trydens::Array64,eta=eta::Float64,U=U::Float64,W=W::Float64,dummf=dummf::Array{Float64,2},cve=cve::Float64,rho=rho::Float64)
  eta=guessfun(d=d,gamma=gamma,cve=cve,rho=rho)
  r=-repn[1]+1
  while r<=repn[2]
      tryun=jumpfun(d=d,gamma=gamma,cve=cve,rho=rho)
      #trydens=exp(myfun(d=d,gamma=gamma,eta=tryun,gvec=gvec,dummf=dummf)*gamma-myfun(d=d,gamma=gamma,eta=eta,gvec=gvec,dummf=dummf)*gamma)
      #dum=rand(n).<trydens
      ##rho dissappears as we pick one that does not change with d,
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
      #trydens=exp(myfun(d=d,gamma=gamma,eta=tryun,gvec=gvec,dummf=dummf)*gamma-myfun(d=d,gamma=gamma,eta=eta,gvec=gvec,dummf=dummf)*gamma)
      #dum=rand(n).<trydens
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

## This function wraps up gavg for parallelization, here it is just a wrap-up.
@everywhere function dvecf(;d=d::Float64,gamma=gamma::Float64,myfun=myfun::Function,guessfun=guessfun::Function,jumpfun=jumpfun::Function,repn=repn,a=a::Array{Float64,2},gvec=gvec::Array{Float64,2},tryun=tryun::Array{Float64,2},trydens=trydens::Array64,eta=eta::Float64,U=U::Float64,W=W::Float64,dummf=dummf::Array{Float64,2},cve=cve::Float64,rho=rho::Float64)
  #etadum=guessfun(d=d,gamma=gamma)
  #@eval @everywhere eta=$etadum
  dvec0= @sync @parallel (+) for i=1:nprocs0
    gavg(d=d,gamma=gamma,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
  end
  dvec0/nprocs0
  #gavg(d=d,gamma=gamma,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
end

## This function wraps up gavraw for parallelization, here it is just a wrap-up.
@everywhere function dgavf(;d=d::Float64,gamma=gamma::Float64,myfun=myfun::Function,guessfun=guessfun::Function,jumpfun=jumpfun::Function,repn=repn,a=a::Array{Float64,2},gvec=gvec::Array{Float64,2},tryun=tryun::Array{Float64,2},trydens=trydens::Array64,eta=eta::Float64,U=U::Float64,W=W::Float64,dummf=dummf::Array{Float64,2},cve=cve::Float64,rho=rho::Float64)
  #etadum=guessfun(d=d,gamma=gamma)
  #@eval @everywhere eta=$etadum
  #vardum=zeros(ltt2+T,ltt2+T)
  #@eval @everywhere vardum=$vardum

  #  dum=gavraw(d=d,gamma=gamma,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,cve=cve,rho=rho)
    #for i=1:n0
    #  vardum=vardum+dum[i,:]*dum[i,:]'
    #end
  #end
  dvec0= @sync @parallel (+) for i=1:nprocs0
    gavraw(d=d,gamma=gamma,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
  end
  dvec0/nprocs0
  #gavraw(d=d,gamma=gamma,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
end

## Specify the β
@everywhere beta=1
## Specify the system tolerance for the optimization step in NLopt
@everywhere toluser=1e-3

tic()
#results= @parallel (vcat) for ind in 1:18
#results= @parallel (vcat) for ind in 1:1
for ind=1:1
  ##Test at d_underline=.1, 2% relative error.
  #deltat=[0.1 0.15 .2 .25 .3 .35 .4 .45 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95][ind]
  deltat=[.01][ind]
     n0=n



  ##########################################################################
  ##########################################################################
  ## First step GMM
  ## Initialize memory matrices
  cdums=zeros(n,K)
  #dg=T*T+T
  #dg=T*T+T*K
  dg=T*K
  gvec=zeros(n,(dg))
  dummf=zeros(n,T,T)
  U=zeros(n,T)
  #eta=zeros(n,(T+K*T))
  eta=zeros(n,dg)
  W=zeros(n,T,K)
  #Wdiff=zeros(n,T,T)
  dummw0=zeros(n,1,K)
  a=zeros(n,dg)
  tryun=zeros(n,(dg))
  eta=zeros(n,(dg))
  trydens=zeros(n)

  ## Loop for values of δ
  i=1
  gammav0=randn(dg)
  #kdg=T*T
  #kdg=1
  while i<=length(deltat)
    d0=deltat[i]

    function obj2(gamma0::Vector, grad::Vector)
      if length(grad) > 0
      end
      eta=guessfun(d=d0,gamma=gamma0,cve=cve,rho=rho)
      dvec0=dvecf(d=d0,gamma=gamma0,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
      #dvec0=gavg(d=d0,gamma=gamma0,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
      return sum((dvec0).^2)/(n0^2)
    end

    #opt=NLopt.Opt(:LN_NELDERMEAD,ltt2)
    opt=NLopt.Opt(:LN_BOBYQA,dg)
    #NLopt.lower_bounds!(opt,vcat(ones(1:kdg).*-Inf,ones((kdg+1):dg).*-Inf))
    #NLopt.upper_bounds!(opt,vcat(ones(1:kdg).*-Inf,ones((kdg+1):dg).*Inf))
    NLopt.lower_bounds!(opt,vcat(ones(1:dg).*-Inf))
    NLopt.upper_bounds!(opt,vcat(ones(1:dg).*Inf))
    NLopt.xtol_rel!(opt,toluser)
    NLopt.min_objective!(opt,obj2)
    (minf,minx,ret) = NLopt.optimize!(opt, gammav0)
    solv[ind,i]=minf
    #print(i)
    #print(ret)
    #print(minf)
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
    ## This eliminates the moments associated with indicators that are zero.
    #inddummy0[1:kdg]=false
    dvecdum=sum(dum,1)[:,inddummy0]
    #vardum=zeros(dg-kg,dg-kg)
    vardum=zeros(dg,dg)
    for j=1:n0
      vardum=vardum+dum[j,inddummy0]*dum[j,inddummy0]'
    end
    ## Computation of Ω
    vardum2=vardum/n0-(dvecdum/n0)'*(dvecdum/n0)
    ##Find the generalized inverse AG2017
    (Lambda,QM)=eig(vardum2)
    ## numerical zero
    #inddummy=Lambda.>1e-30
    inddummy=Lambda.>0.00001
    #inddummy=Lambda.>1e-30

    function obj2(gamma0::Vector, grad::Vector)
      if length(grad) > 0
      end
      eta=guessfun(d=d0,gamma=gamma0,cve=cve,rho=rho)
      ###Solve
      dum=dgavf(d=d0,gamma=gamma0,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
      inddummy0=zeros(dg).<ones(dg)
      #inddummy0[1:kdg]=false
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
    # NLopt.lower_bounds!(opt,vcat(ones(1:kdg).*-Inf,ones((kdg+1):dg).*-Inf))
    # NLopt.upper_bounds!(opt,vcat(ones(1:kdg).*-Inf,ones((kdg+1):dg).*Inf))
    NLopt.lower_bounds!(opt,vcat(ones(1:dg).*-Inf))
    NLopt.upper_bounds!(opt,vcat(ones(1:dg).*Inf))
    NLopt.xtol_rel!(opt,toluser)
    NLopt.min_objective!(opt,obj2)
    (minf,minx,ret) = NLopt.optimize!(opt, gammav0)
    solvw[ind,i]=minf*2*n0
    #print(i)
    #print(ret)
    #print(minf*2*n0)
    solvwgamma[ind,i,:]=minx
    gammav0=randn(dg)
    i=i+1
  end


  #print(solvw[ind,:])
#end
  #print(ind)
  #hcat(solv[ind,:],solvw[ind,:,:],solvgamma[ind,:,:],solvwgamma[ind,:,:])
  #hcat(deltat,solv[ind,:],solvw[ind,:,:],solvgamma[ind,:,:],solvwgamma[ind,:,:])
  results=hcat(solv[1,:],solvw[1,:],solvgamma[1,:,:],solvwgamma[1,:,:])
end
toc()
#AGsfinal=hcat(deltat,AGs[:,1],AGs[:,2])
print(solv[1,:])

print(solvw[1,:])

results


###END simulation
#########################################################################
## Export
DFsolv=convert(DataFrame,results)
writetable(dirresults*"/results_experimental.csv",DFsolv)
#DFsolvw=convert(DataFrame,solvw)
#writetable(dirresults*"/solvwbetaeq1_err5.csv",DFsolvw)
##########################################################################
##########################################################################
