###Author: Victor H. Aguiar and Nail Kashaev
###email: vhaguiar@gmail.com
## Version: JULIA 1.1.0 (2019-01-21)

################################################################################
## Loading Packages and setting up procesors
using Distributed
count = 0
nprocsdum=1
addprocs(nprocsdum)
@everywhere using LinearAlgebra
@everywhere using Random
@everywhere using MathProgBase
@everywhere using Clp
@everywhere using DataFrames
using JuMP
using Ipopt
using CSV
using NLopt
using BlackBoxOptim
using Sobol
using SoftGlobalScope
###############################################################################
## Simulation
beta=.7
npower=1000
Resultspower=DataFrame(hcat(ones(npower).*beta,zeros(npower)))
names!(Resultspower,Symbol.(["beta","TSGMMcueMC"]))
#ri=1;
@softscope for ri=1:2
#while ri<=2
#    global ri+=1
    beta=.7
    beta=beta
    ## Theta

    theta0=.99

    ################################################################################
    ## Setting-up directory
    #rootdir="D:/Dropbox/AKsource/AKEDapp"
    computer="office"
    if computer=="laptop"
        rootdir="D:/Dropbox/AKsource/AKEDapp"
    end
    if computer=="office"
        rootdir="D:/Dropbox/Dropbox/AKsource/AKEDapp"
    end


    ################################################################################
    ##
    # data size
    ##seed
    dataapp="couples"
    @everywhere Random.seed!(12)
    ## sample size
    #singles
    if dataapp=="singles"
        @everywhere   n=185
    end

    if dataapp=="couples"
        @everywhere   n=2004
    end
    ## time length
    @everywhere  T=4
    ## number of goods
    @everywhere  K=17
    # repetitions for the simulation
    ## because the simulations are done using parallel Montecarlo we have 100*nprocs draws.
    @everywhere  repn=(0,50000)

    chainM=zeros(n,T,repn[2])
    ## number of proccesors
     nprocs0=nprocsdum+1

    ###########################################
    @everywhere  dg=T


    ###############################################################################
    ## Data
    ## Price data from Adams et al.

    if dataapp=="singles"
        dir=rootdir*"/singles"
        dirresults=rootdir*"/power/results"

        dum0=CSV.read(dir*"/p.csv",datarow=2,allowmissing=:none)
        dum0=convert(Matrix,dum0[:,:])
        dum0=reshape(dum0,n,T,K)
        @eval @everywhere   p=$dum0
        ## Consumption data from Adams et al.
        dum0=CSV.read(dir*"/cve.csv",datarow=2,allowmissing=:none)
        dum0=convert(Matrix,dum0[:,:])
        ##original scale in the dataset
        dum0=reshape(dum0,n,T,K)
        @eval @everywhere   cve=$dum0
        #sum(ones(n,T,K).*cve,dims=3)

        ## Interest data from Adams et al.
        dum0=CSV.read(dir*"/rv.csv",datarow=2,allowmissing=:none)
        dum0=convert(Matrix,dum0[:,:])
        ## This step is done following the replication code in Adams et al.
        @eval @everywhere  rv=$dum0.+1


    end;

    if dataapp=="couples"
        dir=rootdir*"/couples"
        dirresults=rootdir*"/power/results"
        dum0=CSV.read(dir*"/pcouple.csv",allowmissing=:none)
        dum0=convert(Matrix,dum0[:,:])
        dum0=reshape(dum0,n,T,K)
        @eval @everywhere  p=$dum0
        # consumption array
        dum0=CSV.read(dir*"/cvecouple.csv",allowmissing=:none)
        dum0=convert(Matrix,dum0[:,:])
        #dum0=reshape(dum0,n,T,K)./1e5
        dum0=reshape(dum0,n,T,K)
        @eval @everywhere  cve=$dum0

        # interest rate array
        dum0=CSV.read(dir*"/rvcouple.csv",allowmissing=:none)
        dum0=convert(Matrix,dum0[:,:])
        @eval @everywhere  rv=$dum0.+1


    end;


    ###############################################################################
    ## Data Cleaning
    @everywhere  rho=zeros(n,T,K)

    ## Discounted prices
    @everywhere for i=1:n
      for t=1:T
        rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
      end
    end

    print("load data ready!")








    ################################################################################
    ## Initializing

    Random.seed!(123)


    ################################################################################
    ## Main functions loading and initialization

    ################################################################################
    ## moments
    ## Moment: my function
    include(rootdir*"/restud_apppower/cpufunctions/myfun.jl")
    ## chain generation with CUDA
    include(rootdir*"/restud_apppower/cudafunctions/cuda_chainfun.jl")
    ## optimization with CUDA
    numblocks = ceil(Int, n/167)
    nfast=10000
    Random.seed!(123)
    indfast=rand(1:repn[2],nfast)
    indfast[1]=1
    #indfast=repn[2]-nfast+1:repn[2]
    #indfast=1:nfast
    chainMcu=nothing
    GC.gc()
    chainMcu=cu(chainM[:,:,indfast])
    include(rootdir*"/restud_apppower/cudafunctions/cuda_fastoptim.jl")
    print("functions loaded!")

    #############################################################################
    ## Data generation
    include("D:/Dropbox/Dropbox/AKsource/AKEDapp/restud_apppower/powerfunctions/hyperbolicgen.jl")


    ## warmstart
    include(rootdir*"/restud_apppower/cpufunctions/warm_start_searchdelta.jl")
    print("warm start ready!")


    Random.seed!(123)
    gammav0=zeros(dg)
    gchaincu!(theta0,gammav0,cve,rho,chainM)
    print("chain ready!")


    ###########################################################################3
    ################################################################################################
    ## Optimization step in cuda
    Random.seed!(123)
    indfast=rand(1:repn[2],nfast)
    indfast[1]=1
    chainMcu[:,:,:]=cu(chainM[:,:,indfast])


    ###############################################################################
    ###############################################################################
    Random.seed!(123)
    res = bboptimize(objMCcu2; SearchRange = (-10e300,10e300), NumDimensions = 4,MaxTime = 100.0, TraceMode=:silent)
    #res = bboptimize(objMCcu2; SearchRange = (-10e5,10e5), NumDimensions = 4,MaxTime = 100.0,Method=:probabilistic_descent)


    minr=best_fitness(res)
    TSMC=2*minr*n
    TSMC
    guessgamma=best_candidate(res)


    ###############################################################################
    ###############################################################################
    ## objMCcu(ones(T)*0,ones(T)*0)
    #opt=NLopt.Opt(:LN_NELDERMEAD,dg)
    opt=NLopt.Opt(:LN_BOBYQA,dg)
    toluser=1e-6
    #NLopt.lower_bounds!(opt,ones(dg).*(1e10)*(-1))
    #NLopt.upper_bounds!(opt,ones(dg).*1e10)
    NLopt.lower_bounds!(opt,ones(dg).*-Inf)
    NLopt.upper_bounds!(opt,ones(dg).*Inf)
    NLopt.xtol_rel!(opt,toluser)
    NLopt.min_objective!(opt,objMCcu)
    gammav0=randn(dg)*1000
    #gammav0[:]=gamma1
    (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
    TSMC=2*minf*n
    TSMC
    solvegamma=minx



    #############################################################
    ###########################################



    Resultspower[ri,2]=TSMC
    CSV.write(dirresults*"/power_cuda_$beta.csv",Resultspower)

end;
