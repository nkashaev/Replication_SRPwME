#Version Julia "1.1.1"
using LinearAlgebra
using Random
using MathProgBase
using Clp
using DataFrames
using CSV
################################################################################
## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="FirstApp"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all"
dirdata=repdir*"/Data_all"
################################################################################
## Data
# Singles
# Sample size
const n=185
# Number of time periods
const T=4
# Number of goods
const K=17
###############################################################################
#Prices
dum0=CSV.read(dirdata*"/p.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T,K)
@eval  const p=$dum0

## Consumption
dum0=CSV.read(dirdata*"/cve.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
dum0=reshape(dum0,n,T,K)
@eval  const cve=$dum0

## Interest rates
dum0=CSV.read(dirdata*"/rv.csv",datarow=2,allowmissing=:none)
dum0=convert(Matrix,dum0[:,:])
@eval const rv=$dum0.+1

## Discounted prices
rho=zeros(n,T,K)
for i=1:n
  for t=1:T
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
  end
end
############################################################################
############################################################################
## Constraints
ind=100
q=cve[ind,:,:]'
sol = linprog([-1,0],[2 1],'<',1.5, ClpSolver())
solsuccess=sol.status
stepdum= .03
deltat=collect(.1:stepdum:1)
soldet=zeros(size(deltat,1),n)

function lccons(;delta=delta,p=p,q=q,A=A,b=b)
  for i=1:size(q,2)
  tmp1=-Matrix{Float64}(I, size(q,2), size(q,2))
  tmp1[:,i]=ones(size(q,2),1)
  tmp1[i,:]=zeros(1,size(q,2))
  tmp2=zeros(size(q,2),1)
    for j=1:size(q,2)
      tmp2[j,1]=(delta^(-j+1)*(p[:,j]'*(q[:,i]-q[:,j])))[1]
    end
  A[i,:,:]=tmp1
  b[i,:,:]=tmp2
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

    sol = linprog(c,Ar,'<',br[:,1],lb,ub,ClpSolver())
    soldet[dd,ind]=sol.status==solsuccess
  end
end
## Rate of Rejection
rate=1-sum((sum(soldet,dims=1)[:].>=ones(n))*1)/n
rateDF=convert(DataFrame,["ed-singles" rate])
CSV.write(diroutput*"/AK_FirstApp_singles_deterministic_test.csv",rateDF)

################################################################################
################################################################################

## The code below is almost identical to the code above. The main difference is that it uses data for couples
##Data
# Couples
# Sample size
const n=2004
# Number of time periods
const T=4
# Number of goods
const K=17
###############################################################################
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
############################################################################
############################################################################
## Constraints
ind=100
q=cve[ind,:,:]'
sol = linprog([-1,0],[2 1],'<',1.5, ClpSolver())
solsuccess=sol.status
soldet=zeros(size(deltat,1),n)

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

    sol = linprog(c,Ar,'<',br[:,1],lb,ub,ClpSolver())
    soldet[dd,ind]=sol.status==solsuccess
  end
end
## Rate of Rejection
rate_couples=1-sum((sum(soldet,dims=1)[:].>=ones(n))*1)/n
rate_couplesDF=convert(DataFrame,["ed-couples" rate_couples])
CSV.write(diroutput*"/AK_FirstApp_couples_deterministic_test.csv",rate_couplesDF)
