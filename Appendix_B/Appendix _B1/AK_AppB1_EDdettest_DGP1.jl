using LinearAlgebra
using Random
using MathProgBase
using Clp
using DataFrames
using CSV
## Theta
beta=1
theta0=0.8
npower=1
Resultspower=DataFrame(hcat(ones(npower).*beta,zeros(npower)))
names!(Resultspower,Symbol.(["beta","TSGMMcueMC"]))
################################################################################
## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="Appendix_B/Appendix_B1"
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
@time for ri=1:npower
  Random.seed!(123*ri)
  #############################################################################
  #Generating random drow
  dum0=CSV.read(dirdata*"/pcouple.csv",allowmissing=:none)
  indsim=rand(1:2004,n)
  dum0=convert(Matrix,dum0[indsim,:])
  dum0=reshape(dum0,n,T,K)
  @eval p=$dum0
  # # consumption array
  # dum0=CSV.read(dirdata*"/cvecouple.csv",allowmissing=:none)
  # dum0=convert(Matrix,dum0[indsim,:])
  # dum0=reshape(dum0,n,T,K)
  # @eval cve=$dum0
  # interest rate array
  dum0=CSV.read(dirdata*"/rvcouple.csv",allowmissing=:none)
  dum0=convert(Matrix,dum0[indsim,:])
  @eval rv=$dum0.+1
  ## Data Cleaning
  rho=zeros(n,T,K)
  ## Discounted prices
  for i=1:n
    for t=1:T
      rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
    end
  end
  rhoold=rho
  #print("load data ready!")
  ##############################################################################

  dlow=theta0
  deltasim=rand(n).*(1-dlow).+dlow
  #lambda=ones(n,1)
  lambda=randexp(n)/1
  #lambda=ones(n)
  #psit=hcat(hcat(ones(n).*1,ones(n).*100),hcat(ones(n).*200,ones(n).*300))
  #psit=hcat(hcat(ones(n).*1,ones(n).*2),hcat(ones(n).*1,ones(n).*2))
  su=100
  sl=1/15
  sigma=rand(n,K)*(su-sl) .+ sl

  ##Multiplicative Error
  adum=0.97
  bdum=1.03
  #adum=1.0
  #bdum=1.0
  epsilon=adum .+ rand(n,T,K)*(bdum-adum)
  cve=zeros(n,T,K)
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

      ##This works
      sol = linprog(c,Ar,'<',br[:,1],lb,ub,ClpSolver())
      soldet[dd,ind]=sol.status==solsuccess
      print(sol.status)
    end
  end
  ## Rate of Rejection or not consistent
  rate=1-sum((sum(soldet,dims=1)[:].>=ones(n))*1)/n
  Resultspower[ri,2]=rate
  CSV.write(diroutput*"/deter_n_$n._theta_$theta0._beta_$beta.csv",Resultspower)
  GC.gc()
end
