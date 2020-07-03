#Version Julia "0.5.0"
#Author: Victor H. Aguiar
#email: vhaguiar@gmail.com

count = 0
#Set the number of processors: Change it to the max the computer allows
nprocs=16
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
burnrate=4
nsimsp=60
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
#@everywhere include($rootdir*"/AK_myfunc_gabaix.jl")

## AK_myfunc_tremblinghand.jl tests for E[w^c]=0 or E[c^*]=E[c].
include(rootdir*"/AK_myfunc_tremblinghand.jl")


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

#@everywhere include($rootdir*"/AK_guessfunc_priceexperimental.jl")


## Here it invokes the revealedPrefsmod function simGarpQuantWealth, that will generate a draw of p^* that satisfies GARP and in on the budget line.
## The function allows to set an afriatpar that corresponds to the cost efficiency index. We set it to 1.
#maxit is the max. number of iterations allowed by the sampler before it restarts.
#R has to get a random seed.
#Do not pay attention to the name of the files cvesim since it does not matter, in this case it is filled by prices
include(rootdir*"/AK_guessfunc_quantityexperimental.jl")



###############################################################
## New Fast jump
## This function will draw new candidates for the Montecarlo, in this case this is the same as the guessfun.
## The reason is that in this case, we can generate exactly data under the null of GARP plus being on the budget.

## For prices
#@everywhere include($rootdir*"/AK_jumpfunc_priceexperimental.jl")

## For quantities
include(rootdir*"/AK_jumpfunc_quantityexperimental.jl")


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

#opt=NLopt.Opt(:LN_NELDERMEAD,ltt2+T)
opt=NLopt.Opt(:LN_BOBYQA,dg)
NLopt.lower_bounds!(opt,vcat(ones(dg).*-Inf))
NLopt.upper_bounds!(opt,vcat(ones(dg).*Inf))
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,obj2)
gammav0=randn(dg)
(minf,minx,ret) = NLopt.optimize!(opt, gammav0)
solvw[ind,i]=minf*2*n
solvwgamma[ind,i,:]=minx
gammav0=randn(dg)

##Iter 2
(minf,minx,ret) = NLopt.optimize!(opt, minx)
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
CSV.write(dirresults*"/results_experimental_quantity.csv",DFsolv)
#DFsolvw=convert(DataFrame,solvw)
#writetable(dirresults*"/solvwbetaeq1_err5.csv",DFsolvw)
##########################################################################
##########################################################################


gammaopt=[-1.048351165	0.063172365	1.543420659	-0.614591743	-0.73938166	-0.18236565	-1.489497893	-0.429092051	0.243673898	-1.825550162	1.686882104	1.955137095	0.321053245	-0.466084768	-3.76200587	1.070628763	-0.739000561	-0.987438343	1.250501322	1.559541424	-0.363714918	-1.47960846	-0.0385697	1.038161698	-0.661628464	-1.452147984	1.710400633	1.00134395	-1.390226524	1.895356167	0.332357628	0.31278488	-0.612672827	1.852387256	-0.113466695	-0.735835497	0.567418687	0.867046951	0.825432474	-0.500824617	0.634994639	1.08791278	2.407399106	-0.651285641	-0.635520331	0.360076374	-0.82390626	0.060121114	-0.171351508	-0.857042168	-1.302939941	-2.684976164	-1.054671293	-0.451590789	1.498873115	0.323282482	0.546206037	-0.208151862	-1.137207103	2.082374716	0.809629323	0.761512065	-0.214177601	1.056879144	1.964261931	-0.344045379	-0.984289744	0.696144687	-0.765131292	-1.249043008	-0.944383801	-1.281671331	-0.905531975	0.581299377	0.517677722	-0.678833526	-1.09231771	1.314711142	0.176678826	-0.775801359	0.500579499	-0.113358755	-0.617343253	-0.943462831	0.241397859	0.681867066	0.903663615	-0.81847091	0.325782623	0.636629792	0.06216887	-0.574725224	0.329659055	0.67990876	-0.029610793	-0.189411902	-0.238805204	-0.446383016	-0.713873501	-0.390233016	0.706534351	0.474725683	-0.688972577	0.050701701	0.449932653	1.259894482	1.099094841	-0.530270891	1.556247971	-0.000834987	0.331157042	1.286255288	-0.049496623	-0.046025776	-2.736760473	0.052188056	1.637252427	-0.754317617	-0.803553612	-0.061114673	-0.276596074	0.727330795	-0.201472714	1.025558457	0.693388885	-1.047992745	-0.652765961	0.281794534	-0.003837917	-0.459791636	1.244317918	-0.037589855	1.185828658	1.452814608	-1.792405095	-0.619904036	0.279426853	-0.935928697	1.563590491	-0.816702798	0.010163766	-0.623097233	-0.289783438	-1.027543471	0.078247642	1.240494305	0.569913949	-1.003629364	0.263405973	-0.766266122	-0.907697098	-0.599161258	1.230741106	0.938988126	0.000692907	0.150039686	-0.520901959	-1.511228532	0.335293581	-1.292723335	0.858432975	0.583321621	1.520698167	0.279708982	-1.1113114	1.001240339	1.274467863	1.665882788	-0.694792583	-0.897955439	2.57220503	-0.100484517	1.845139708	0.308044446	-0.769236457	0.602347558	0.71855213	-0.325556074	-0.964930622	-0.62838337	-0.014939583	1.225622905	-1.626961644	-1.000325334	0.753670077	-1.628547517	-0.172308205	-1.901172553	-1.162235841	-0.254997596	0.897732916	0.113206791	-0.260854404	2.353732454	0.590853302	0.159517597	1.40824129	-0.330329325	-0.793018786	-0.538504568	0.401813508	-1.440686927	-0.13178537	0.711103118	-1.815423647	-1.479670125	3.342892084	0.187593961	0.279531544	0.089874177	-0.338200452	1.351075211	0.315450364	0.683238923	-0.336075832	-0.089162968	0.915996476	-2.484786226	-0.856018197	-0.264486578	-0.507591243	0.717865609	-0.090619147	0.906077946	-0.864975043	0.553981421	0.353524268	0.599756713	-0.88443235	-0.930990764	-0.351254521	1.323674609	-0.221617423	-0.874429829	0.954687355	-0.752243053	0.453400207	-0.419106496	-1.431432146	0.058622142	-0.576366761	0.973640757	-1.521089148	1.540965991	2.006978657	-0.230581566	-0.241886224	-0.524620975	-1.438268371	-2.194463522	-0.540810781	0.490148548	-1.206478585	0.402922581	-0.65607279	-1.165871794	-0.126330109	1.256418948	-1.787463123	-0.480146661	0.656089801	-0.135753747	0.480449744	-0.610060801	-0.779640739	-2.36786624	-0.517556834	-0.66088915	2.308221515	0.394804547	1.90748293	-0.399379517	-0.512163641	0.527548302	0.291874713	-0.335517356	0.458759716	0.819786545	1.707647279	-0.47702087	-0.511646748	0.111520362	0.567388241	-0.153771613	-0.178768087	-0.188895571	-0.001531865	-0.842968956	-1.370793295	-1.552626517	-1.905329582	0.395002323	0.095497991	0.9609055	-0.456232013	-0.450081704	0.379421381	0.180914324	-1.330939624	-0.070382932]
gammaopt2=gammaopt[151:300]
obj2(gammaopt2,[0])


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
