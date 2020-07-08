using LinearAlgebra
using Random
using MathProgBase
using Clp
using DataFrames
using CSV
## Theta
npower=1000
Resultspower=DataFrame(hcat(ones(npower).*1.0,zeros(npower)))
names!(Resultspower,Symbol.(["seed","RejRate"]))
################################################################################
## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="Appendix_B"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all/Appendix"
dirdata=repdir*"/Data_all"
################################################################################
## Data
# Singles
# Sample size
const n=2000
# Number of time periods
const T=4
# Number of goods
const K=17
##
PP=CSV.read(dirdata*"/pcouple.csv",allowmissing=:none)
RR=CSV.read(dirdata*"/rvcouple.csv",allowmissing=:none)
for ri=1:npower
Random.seed!(123*ri)
#############################################################################
#Generating random data
#Prices
indsim=rand(1:2004,n)
dum0=convert(Matrix,PP[indsim,:])
dum0=reshape(dum0,n,T,K)
@eval p=$dum0
# Interest rates
dum0=convert(Matrix,RR[indsim,:])
@eval rv=$dum0.+1
## Discounted prices
rho=zeros(n,T,K)
for i=1:n
  for t=1:T
    rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
  end
end
##############################################################################
##Generating consumption using DGP1
cve=zeros(n,T,K)
dlow=1.0
deltasim=rand(n).*(1-dlow).+dlow
lambda=randexp(n)/1
psit=hcat(hcat(ones(n).*2,ones(n).*1),hcat(ones(n).*2,ones(n).*1))
su=100
sl=1/15
sigma=rand(n,K)*(su-sl) .+ sl
##Multiplicative Error
adum=0.97
bdum=1.03
epsilon=adum .+ rand(n,T,K)*(bdum-adum)
@simd for i=1:n
   for t=1:T
     for k=1:K
       cve[i,t,k]= ((lambda[i]/deltasim[i]^(t-1))*rho[i,t,k]).^(-1/sigma[i,k])*epsilon[i,t,k]
     end
   end
 end

############################################################################
############################################################################
## Testing DGP 1
ind=100
q=cve[ind,:,:]'
sol = linprog([-1,0],[2 1],'<',1.5, ClpSolver())
solsuccess=sol.status
stepdum= .05
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

## Rejection Rate
rate=1-sum((sum(soldet,dims=1)[:].>=ones(n))*1)/n
Resultspower[ri,2]=rate
CSV.write(diroutput*"/deter_null_theta0_$dlow._n_$n.csv",Resultspower)
GC.gc()
end
