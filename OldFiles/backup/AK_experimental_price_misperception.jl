#Version Julia "0.5.0"
#Author: Victor H. Aguiar
#email: vhaguiar@gmail.com

count = 0
#Set the number of processors: Change it to the max the computer allows
nprocs=32
using Distributed
addprocs(nprocs)
@everywhere Distributed
@everywhere using Random
# Set a random seed
@eval @everywhere vern=6
@everywhere Random.seed!(3000)
@everywhere using NLopt
@everywhere using DataFrames
@everywhere using MathProgBase
using CSV
##the following line is eliminated in case the installation of Rcall happened correctly and there is only 1 installation of R
@everywhere ENV["R_HOME"]="C:\\Users\\Nkashaev\\Documents\\R\\R-3.4.4"

@everywhere using RCall
@everywhere using LinearAlgebra
## set directory
#Dusky cluster dir:
#rootdir="/home/vaguiar/AKsims"
## Office computer
rootdir="C:/Users/Nkashaev/Dropbox/ReplicationAK/SecondApp"

dir=rootdir*"/data"

#results file
dirresults=rootdir*"/results"

# data size
@everywhere  n=154

## time length
@everywhere const T=50
#@everywhere const T=4
## number of goods
@everywhere const K=3
# repetitions for the simulation
## because the simulations are done using parallel Montecarlo we have 100*nprocs draws.
# set burn
burnrate=2
nsimsp=30
@everywhere const repn=($burnrate,$nsimsp)
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

## myfunc_gabaix.jl tests for E[w^p]=0 or E[p^*]=E[p].
include(rootdir*"/AK_myfunc_gabaix.jl")

## AK_myfunc_tremblinghand.jl tests for E[w^c]=0 or E[c^*]=E[c].
#include(rootdir*"/AK_myfunc_tremblinghand.jl")


##New Guess Functions: Constraint to satisfy pw=0 as.
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

include(rootdir*"/AK_guessfunc_priceexperimental.jl")


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
include(rootdir*"/AK_jumpfunc_priceexperimental.jl")

## For quantities
#include(rootdir*"/AK_jumpfunc_quantityexperimental.jl")


## The Montecarlo step: It gives the integrated moments h
## This code follows Schennach's code in Gauss in the Supplement in ECMA for ELVIS.
@everywhere function gavg(;d=d::Float64,gamma=gamma::Float64,myfun=myfun::Function,guessfun=guessfun::Function,jumpfun=jumpfun::Function,repn=repn,a=a::Array{Float64,2},gvec=gvec::Array{Float64,2},tryun=tryun::Array{Float64,2},trydens=trydens::Array64,eta=eta::Float64,U=U::Float64,W=W::Float64,dummf=dummf::Array{Float64,2},cve=cve::Float64,rho=rho::Float64)
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
    sum(a,dims=1)/repn[2]
end


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

## This function wraps up gavg for parallelization, here it is just a wrapper.
@everywhere function dvecf(;d=d::Float64,gamma=gamma::Float64,myfun=myfun::Function,guessfun=guessfun::Function,jumpfun=jumpfun::Function,repn=repn,a=a::Array{Float64,2},gvec=gvec::Array{Float64,2},tryun=tryun::Array{Float64,2},trydens=trydens::Array64,eta=eta::Float64,U=U::Float64,W=W::Float64,dummf=dummf::Array{Float64,2},cve=cve::Float64,rho=rho::Float64)
  dvec0= @sync @distributed (+) for i=1:nprocs0
    gavg(d=d,gamma=gamma,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
  end
  dvec0/nprocs0
end

## This function wraps up gavraw for parallelization, here it is just a wrapper.
@everywhere function dgavf(;d=d::Float64,gamma=gamma::Float64,myfun=myfun::Function,guessfun=guessfun::Function,jumpfun=jumpfun::Function,repn=repn,a=a::Array{Float64,2},gvec=gvec::Array{Float64,2},tryun=tryun::Array{Float64,2},trydens=trydens::Array64,eta=eta::Float64,U=U::Float64,W=W::Float64,dummf=dummf::Array{Float64,2},cve=cve::Float64,rho=rho::Float64)
  dvec0= @sync @distributed (+) for i=1:nprocs0
    gavraw(d=d,gamma=gamma,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
  end
  dvec0/nprocs0
end

## Specify the system tolerance for the optimization step in NLopt, set to 1e-6, for speed 1e-3 seems to be doing the same
@everywhere toluser=1e-3

#tic()

  #Set a line search over the lowerbounds of the discount factor
  # In this case this is innesential so set it up to anything.
  #deltat=[0.1 0.15 .2 .25 .3 .35 .4 .45 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95][ind]



##########################################################################
##########################################################################
## First step GMM
## Initialize memory matrices
cdums=zeros(n,K)
#  dg=T*K
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
d0=1.0

function obj0(gamma0::Vector, grad::Vector)
      if length(grad) > 0
      end
      eta=guessfun(d=d0,gamma=gamma0,cve=cve,rho=rho)
      dvec0=dvecf(d=d0,gamma=gamma0,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
      return sum((dvec0).^2)/(n^2)
end

    #Both LN_NELDERMEAD and LN_BOBYQA do the work, but the second is the fastest altenative derivative-free
    #opt=NLopt.Opt(:LN_NELDERMEAD,ltt2)
opt=NLopt.Opt(:LN_BOBYQA,dg)
NLopt.lower_bounds!(opt,vcat(ones(dg).*-Inf))
NLopt.upper_bounds!(opt,vcat(ones(dg).*Inf))
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,obj0)
(minf,minx,ret) = NLopt.optimize!(opt, gammav0)
solv[ind,i]=minf
solvgamma[ind,i,:]=minx
gammav0=randn(dg)



  #########################################################################
  ## Weighted Objective: Second Step GMM
  ## Loop for different values of δ
gammav0=randn(dg)
i=1
ind=1
d0=1

# function obj2(gamma0::Vector, grad::Vector)
#       if length(grad) > 0
#       end
#       cdums=zeros(n,K)
#       #  dg=T*K
#       gvec=zeros(n,(dg))
#       dummf=zeros(n,T,T)
#       U=zeros(n,T)
#       eta=zeros(n,dg)
#       W=zeros(n,T,K)
#       dummw0=zeros(n,1,K)
#       a=zeros(n,dg)
#       tryun=zeros(n,(dg))
#       eta=zeros(n,(dg))
#       trydens=zeros(n)
#       d0=1.0
#
#       eta=guessfun(d=d0,gamma=gamma0,cve=cve,rho=rho)
#       ###Solve
#       dvecM=dgavf(d=d0,gamma=gamma0,myfun=myfun,guessfun=guessfun,jumpfun=jumpfun,repn=repn,a=a,gvec=gvec,tryun=tryun,trydens=trydens,eta=eta,U=U,W=W,dummf=dummf,cve=cve,rho=rho)
#       dvec=sum(dvecM,dims=1)'/n
#
#
#         numvar=zeros(dg,dg)
#         @simd for i=1:n
#             BLAS.syr!('U',1.0/n,dvecM[i,:],numvar)
#         end
#         var=numvar+numvar'- Diagonal(diag(numvar))-dvec*dvec'
#         (Lambda,QM)=eigen(var)
#         inddummy=Lambda.>0.00001
#         An=QM[:,inddummy]
#         dvecdum2=An'*(dvec)
#         vardum3=An'*var*An
#         Omega2=inv(vardum3)
#         Qn2=1/2*dvecdum2'*Omega2*dvecdum2
#
#         return Qn2[1]
# end

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
#opt=NLopt.Opt(:LN_NELDERMEAD,ltt2+T)

gammaopt=[-1.051510504	0.063194295	1.54519104	-0.610418741	-0.74098141	-0.18216824	-1.492549872	-0.425557165	0.242696908	-1.821774032	1.680772002	1.959792269	0.31977132	-0.468314342	-1.894518935	1.068180089	-0.740375902	-0.98148005	1.251343206	1.554009563	-0.363614015	-1.475531997	-0.038790218	1.039918772	-0.658935354	-1.457490625	1.711825487	1.005949502	-1.38732698	1.890405786	0.332699592	0.309375332	-0.612017253	1.846765671	-0.113912111	-0.733425284	0.566272127	0.869617034	0.823502613	-0.500787826	0.640252451	1.098864178	2.379419936	-0.647059565	-0.628339195	0.356710277	-0.817570893	0.060059253	-0.171357571	-0.852158361	-1.285736055	-2.676690711	-1.053816993	-0.443834234	1.496285905	0.324503461	0.540442927	-0.207519913	-1.142871176	2.146065172	0.815461152	0.766497241	-0.212276833	1.049420118	1.963268817	-0.344211093	-0.969626289	0.700514045	-0.757260913	-1.247979271	-0.943728632	-1.268422602	-0.90326745	0.56538005	0.515448403	-0.681185892	-1.086477224	1.324467526	0.176445009	-0.77866173	0.491110362	-0.112147705	-0.62398352	-0.946125954	0.239959465	0.685317268	0.93904451	-0.8425762	0.325775039	0.628343863	0.061687511	-0.57773518	0.330544154	0.685434722	-0.02995035	-0.188272993	-0.240745114	-0.447957785	-0.710988296	-0.390029802	0.728748544	0.485406964	-0.687547836	0.050810284	0.4676776	1.260027951	1.103148635	-0.533640836	1.594603535	-0.000838556	0.332988758	1.316609937	-0.049559028	-0.046449082	-2.710061731	0.052168364	1.629713262	-0.759603593	-0.805449541	-0.061289811	-0.278516139	0.728505178	-0.201565097	0.967263841	0.676582543	-1.072491038	-0.653111782	0.281190079	-0.003762111	-0.463424173	1.2511311	-0.037218574	1.178422493	1.453482755	-1.791292951	-0.619683178	0	-0.937770511	1.566552389	-0.78598868	0.010157337	-0.624783025	-0.287206959	-1.021243715	0.077569418	1.258776302	0.571991303	-1.0483388	0.260378706	-0.762581785	-0.749934615	3.154629911	-1.325851498	1.777784854	-0.880099917	-0.436669845	-0.985752968	-2.023902847	0.209956415	-0.57334543	0.20654403	-0.177959145	0.98596891	-0.405004729	-2.229913818	-1.057179353	0.283139137	-0.692339122	0.056342602	-0.482564605	-0.2879306	-0.053176409	2.331567222	1.110850847	-2.053533189	-0.05019977	-0.873020064	-0.486178981	1.494958924	0.055555513	1.915606936	-0.009511438	-0.819106331	-1.180866748	-0.845497157	-0.524942074	-0.146415221	-0.206205983	-0.411524406	-0.323080344	0.487201799	-0.832778019	0.098008845	-0.421059435	-2.174897732	0.871290861	0.925289183	0.225051562	2.053696304	-0.459326193	-0.697589408	1.482146314	-0.091920543	0.297348036	-1.239543907	-0.438463108	0.353274071	-0.604779084	-0.418703863	-0.01104025	-0.125298436	0.881965763	-0.822525274	0.991826011	-0.101576262	0.518903497	0.262160659	-0.080510688	-1.701783533	0.620704398	0.645238912	-0.534586073	-0.143646043	0.756893559	-0.631568327	-2.345215059	0.196888291	2.149342783	0.446152674	0.00122832	0.472625546	-0.101964423	0.847582742	0.106019727	0.921494277	-0.624273423	-0.606745919	-0.028511774	-0.12006135	-0.745580388	-0.22335927	-0.705406771	0.234551524	-1.088039312	0.103636873	-0.718114145	-0.000457179	0.340587048	-0.836371559	0.926260619	0.112965508	0.661302042	0.572308781	2.300316122	1.799373913	-0.127811383	-1.735191696	-1.011267731	-1.217999614	0.201851129	-1.987178722	-0.074487892	-0.744876599	-1.067529739	-0.939992419	-0.664925663	1.287057615	0.673995732	0.794737396	-0.131595577	0.316926596	-0.459181889	-0.772843452	0.959586308	-0.572230025	1.537771402	0.422148792	1.374639188	-0.483800942	-0.454096145	0.205602359	0.171922915	-0.516244094	0.01026002	0.29580913	-0.20173953	-0.252841412	1.222773884	-0.675560577	-0.39129547	-0.140032443	-0.195892018	-0.499058529	1.908083223	-0.272054821	-2.048969024	2.712321405	-0.926388305	-1.346542519	0.650764255]

gammaopt2=gammaopt[151:300]


opt=NLopt.Opt(:LN_BOBYQA,dg)
NLopt.lower_bounds!(opt,vcat(ones(dg).*-Inf))
NLopt.upper_bounds!(opt,vcat(ones(dg).*Inf))
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,obj2)
gammav0=randn(dg)
(minf,minx,ret) = NLopt.optimize!(opt, gammaopt2)
solvw[ind,i]=minf*2*n
solvwgamma[ind,i,:]=minx
gammav0=randn(dg)



results=hcat(solv[1,:],solvw[1,:],solvgamma[1,:,:],solvwgamma[1,:,:])

#toc()
#AGsfinal=hcat(deltat,AGs[:,1],AGs[:,2])
print(solv[1,:])

print(solvw[1,:])

results

#########################################################################
## Export
DFsolv=convert(DataFrame,results)
CSV.write(dirresults*"/results_experimental_price.csv",DFsolv)
#DFsolvw=convert(DataFrame,solvw)
#writetable(dirresults*"/solvwbetaeq1_err5.csv",DFsolvw)
##########################################################################
##########################################################################


obj2(gammaopt2,[0])
2*n*0.0455

function objold(gamma0::Vector, grad::Vector)
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

objold(gammaopt2,[0])
2*n*1.20
