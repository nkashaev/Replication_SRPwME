# ###Author: Victor H. Aguiar and Nail Kashaev
# ###email: vhaguiar@gmail.com
# ## Version: JULIA 1.1.0 (2019-01-21)
#
# ################################################################################
# ## Loading Packages and setting up procesors

using SoftGlobalScope
###############################################################################
## Simulation
beta=1.0
npower=1000
Resultspower=DataFrame(hcat(ones(npower).*beta,zeros(npower)))
names!(Resultspower,Symbol.(["beta","TSGMMcueMC"]))
#ri=1;

@softscope for ri=1:100
    startit=1
    endit=100
    theta0=.975
    @everywhere kap=1.1
    @everywhere target=1
    @everywhere bshare=.07

    ################################################################################
    ## Setting-up directory
    #rootdir="D:/Dropbox/AKsource/AKEDapp"
    computer="lancemachine"
    if computer=="laptop"
        rootdir="D:/Dropbox/AKsource/AKEDapp"
    end
    if computer=="office"
        rootdir="D:/Dropbox/Dropbox/AKsource/AKEDapp"
    end
    if computer=="lancemachine"
        rootdir="C:/Users/nkashaev/Dropbox/AKsource/AKEDapp"
    end


    ################################################################################
    ##
    # data size
    ##seed
    dataapp="couples"
    Random.seed!(123*ri)
    ## sample size


    if dataapp=="couples"
        @everywhere  const n=2000
    end
    ## time length
    @everywhere const T=4
    ## number of goods
    @everywhere const K=17
    # repetitions for the simulation
    ## because the simulations are done using parallel Montecarlo we have 100*nprocs draws.
    @everywhere const repn=(0,10000)

    chainM=zeros(n,T,repn[2])
    ## number of proccesors
    nprocs0=nprocsdum+1

    ###########################################
    @everywhere const dg=T


    ###############################################################################
    ## Data
    ## Price data from Adams et al.

    if dataapp=="couples"
        dir=rootdir*"/couples"
        dirresults=rootdir*"/power/results"
        dum0=CSV.read(dir*"/pcouple.csv",allowmissing=:none)
        indsim=rand(1:2004,n)
        dum0=convert(Matrix,dum0[indsim,:])
        dum0=reshape(dum0,n,T,K)
        @eval @everywhere const p=$dum0
        # consumption array
        dum0=CSV.read(dir*"/cvecouple.csv",allowmissing=:none)
        dum0=convert(Matrix,dum0[indsim,:])
        #dum0=reshape(dum0,n,T,K)./1e5
        dum0=reshape(dum0,n,T,K)
        @eval @everywhere const cve=$dum0

        # interest rate array
        dum0=CSV.read(dir*"/rvcouple.csv",allowmissing=:none)
        dum0=convert(Matrix,dum0[indsim,:])
        @eval @everywhere const rv=$dum0.+1


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

    rhoold=rho

    print("load data ready!")





    ################################################################################
    ## Initializing

    Random.seed!(123*ri)
    gammav0=zeros(dg)




    ################################################################################
    ## Main functions loading and initialization

    ################################################################################
    ## moments
    ## Moment: my function
    include(rootdir*"/restud_apppower/cpufunctions/myfun.jl")
    ## chain generation with CUDA
    include(rootdir*"/restud_apppower/cudafunctions/cuda_chainfun.jl")
    ## optimization with CUDA
    numblocks = ceil(Int, n/100)
    nfast=10000
    Random.seed!(123*ri)
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
    ##Hyperbolic
    #include(rootdir*"/restud_apppower/powerfunctions/hyperbolicgenerator_simulations_2019_9_26.jl")

    ## Collective
    include(rootdir*"/restud_apppower/powerfunctions/couples_generator_simulations_2019_9_28.jl")

    ## warmstart
    include(rootdir*"/restud_apppower/cpufunctions/warm_start_searchdelta_justcvex.jl")
    print("warm start ready!")


    Random.seed!(123*ri)
    gchaincu!(theta0,gammav0,cve,rho,chainM)
    print("chain ready!")


    ###########################################################################3
    ################################################################################################
    ## Optimization step in cuda
    Random.seed!(123*ri)
    indfast=rand(1:repn[2],nfast)
    indfast[1]=1
    chainMcu[:,:,:]=cu(chainM[:,:,indfast])


    ###############################################################################
    ###############################################################################
    Random.seed!(123*ri)
    res = bboptimize(objMCcu2; SearchRange = (-10e300,10e300), NumDimensions = 4,MaxTime = 100.0, TraceMode=:silent)
    #res = bboptimize(objMCcu2; SearchRange = (-10e5,10e5), NumDimensions = 4,MaxTime = 100.0,Method=:probabilistic_descent)


    minr=best_fitness(res)
    TSMC=2*minr*n
    TSMC
    guessgamma=best_candidate(res)


    ###############################################################################
    ###############################################################################


    opt=NLopt.Opt(:LN_BOBYQA,dg)
    toluser=1e-6
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
    guessgamma=solvegamma
    ret
    ##try 2
    if (TSMC>=9)
        (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
        TSMC=2*minf*n
        TSMC
        solvegamma=minx
        guessgamma=solvegamma
        ret
    end
    #try 3
    if (TSMC>=9)
        (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
        TSMC=2*minf*n
        TSMC
        solvegamma=minx
        guessgamma=solvegamma
        ret
    end
########
    Resultspower[ri,2]=TSMC
    CSV.write(dirresults*"/alternateCollective_$computer._power_cuda_start.$startit.end.$endit.csv",Resultspower)
    GC.gc()

end;
