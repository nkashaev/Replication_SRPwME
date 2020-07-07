# using Distributed
# count = 0
# nprocsdum=1
# addprocs(nprocsdum)
using LinearAlgebra
using Random
using MathProgBase
using Clp
using DataFrames
using Statistics
#using JuMP
#using Ipopt
using CSV
#using NLopt
#using BlackBoxOptim
#using Sobol
## Theta
beta=.8
theta0=1

################################################################################
## Setting-up directory
#rootdir="D:/Dropbox/AKsource/AKEDapp"
computer="nailmachine"
if computer=="nailmachine"
    rootdir="/Users/SSC4044-iMac27/Dropbox/AKsource/AKEDapp"
end
if computer=="office"
    rootdir="/Users/SSC4044-iMac27/Dropbox/Dropbox/AKsource/AKEDapp"
end
if computer=="lancemachine"
    rootdir="/Users/SSC4044-iMac27/Dropbox/AKsource/AKEDapp"
end


################################################################################
##
# data size
##seed
dataapp="couples"
Random.seed!(12)
## sample size


if dataapp=="couples"
     n=2004
end
## time length
 T=4
## number of goods
 K=17
# repetitions for the simulation
## because the simulations are done using parallel Montecarlo we have 100*nprocs draws.
 repn=(0,50000)

chainM=zeros(n,T,repn[2])
## number of proccesors
#nprocs0=nprocsdum+1

###########################################
 dg=T


###############################################################################
## Data
## Price data from Adams et al.

if dataapp=="couples"
    dir=rootdir*"/couples"
    #dir=rootdir*"/singles"
    dirresults=rootdir*"/power/results"
    dum0=CSV.read(dir*"/pcouple.csv",allowmissing=:none)
    #dum0=CSV.read(dir*"/rv.csv",allowmissing=:none)
    indsim=rand(1:2004,n)
    dum0=convert(Matrix,dum0[indsim,:])
    #dum0=convert(Matrix,dum0)
    dum0=reshape(dum0,n,T,K)
    @eval p=$dum0
    # consumption array
    dum0=CSV.read(dir*"/cvecouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[indsim,:])
    #dum0=reshape(dum0,n,T,K)./1e5
    dum0=reshape(dum0,n,T,K)
    @eval cve=$dum0

    # interest rate array
    dum0=CSV.read(dir*"/rvcouple.csv",allowmissing=:none)
    dum0=convert(Matrix,dum0[indsim,:])
    @eval rv=$dum0.+1


end;


###############################################################################
## Data Cleaning
 rho=zeros(n,T,K)

## Discounted prices
for i=1:n
  for t=1:T
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
  end
end

rhoold=rho

print("load data ready!")




dlow=theta0
deltasim=rand(n).*(1-dlow).+dlow
lambda=ones(n,1)
lambda=randexp(n)/1
#lambda=ones(n)
#psit=hcat(hcat(ones(n).*1,ones(n).*100),hcat(ones(n).*200,ones(n).*300))
psit=hcat(hcat(ones(n).*1,ones(n).*2),hcat(ones(n).*1,ones(n).*2))
su=100
sl=1/15
sigma=rand(n,K)*(su-sl) .+ sl

##Multiplicative Error
adum=0.97
bdum=1.03
  # adum=1.0
  # bdum=1.0
epsilon=adum .+ rand(n,T,K)*(bdum-adum)
for i=1:n
   for t=1:T
     for k=1:K
       cve[i,t,k]= beta<1 ? ((lambda[i]/deltasim[i]^(t-1))*rho[i,t,k]*psit[i,t]).^(-1/sigma[i,k])*epsilon[i,t,k] : ((lambda[i]/deltasim[i]^(t-1))*rho[i,t,k]).^(-1/sigma[i,k])*epsilon[i,t,k]
     end
   end
 end

#cve=cve*1e7

ind=100
############################################################################
############################################################################
## Constraints
q=cve[ind,:,:]'
sol = linprog([-1,0],[2 1],'<',1.5, ClpSolver())
solsuccess=sol.status
stepdum= .05
deltat=collect(.1:stepdum:1)
soldet=zeros(size(deltat,1),n)
function lccons(;delta=delta,p=p,q=q,A=A,b=b)
  for i=1:size(q,2)
    tmp1=-Matrix{Float64}(I, size(q,2), size(q,2))
    #tmp1=-eye(size(q,2))
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


  A0=zeros(size(q,2),size(q,2),size(q,2))
  b0=zeros(size(q,2),size(q,2),1)
for ind=1:n
  for dd=1:size(deltat,1)
    (Ar,br)=lccons(delta=deltat[dd],p=rho[ind,:,:]',q=cve[ind,:,:]',A=A0,b=b0)

    Ar=reshape(Ar,size(q,2)*size(q,2),size(q,2))
    br=reshape(br,size(q,2)*size(q,2),1)
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

    #sense=['=' '=' '=' '=']'
    #sense=zeros()

    #sol = linprog(c,Ar,'<',br[:,1],lb,ub,ClpSolver())
    ##This works
    sol = linprog(c,Ar,'<',br[:,1],lb,ub,ClpSolver())

    #sol = linprog(c,Ar,'=',br[:,1],lb,ub,ClpSolver())

    #sol = linprog([-1,0],[2 1],'<',1.5, ClpSolver())
    soldet[dd,ind]=sol.status==solsuccess
    print(sol.status)
  end
end
## Rate of Rejection or not consistent
rate=1-sum((sum(soldet,dims=1)[:].>=ones(n))*1)/n
rate
ratev=zeros(1,1)
ratev[:]=rate
DFsolv=convert(DataFrame,ratev)


share=rho.*cve./repeat(sum(rho.*cve,dims=3),1,1,17)
mean(share[:,1,:],dims=1)'
