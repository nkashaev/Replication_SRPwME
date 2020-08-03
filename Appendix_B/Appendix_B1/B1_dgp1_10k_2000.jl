## Loading general packages
using LinearAlgebra
using Random
using MathProgBase
using DataFrames
using CSV
using NLopt
using BlackBoxOptim
# CUDA packages
using CuArrays
using CuArrays.CURAND
using CUDAnative
using CUDAdrv
# warmstart packages
using Convex
using ECOS

using SoftGlobalScope

###############################################################################
## Number of time periods
const T=4
# Number of moment conditions (degrees of freedom)
const dg=4
# Simulation sample size
const n=2000
# Number of goods
const K=17
## MCMC Chain length.
# Burning is optional.
const repn=(0,10000)        #repn=(burn,number_simulations)
chainM=zeros(n,dg,repn[2])  # Initializing MCMC chain
# nfast is the number of draws from eta.
const nfast=10000
# Passing the draws from eta to CUDA.
chainMcu=cu(chainM[:,:,1:nfast])
# Lower bound of the support of the discount factor
theta0=.8

## This function simulates the data and computes the value of TS
function powersimulations(chainM,chainMcu,theta0,n,repn,nfast)
    npower=1000 # Number of simulations.
    # Setting-up the output file
    Resultspower=DataFrame(hcat(ones(npower),zeros(npower)))
    names!(Resultspower,Symbol.(["iter","TSGMMcueMC"]))

    ## Setting-up directory
    tempdir1=@__DIR__
    repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
    appname="Appendix_B"
    rootdir=repdir*"/"*appname
    diroutput=repdir*"/Output_all/Appendix"
    dirdata=repdir*"/Data_all"

    # Sample size of the original data
    nold=2004
    # Number of time periods
    T=4
    # Number of goods
    K=17
    ## Repetitions for the integration step
    dg=T              # dg=degrees of freedom

    ###############################################################################
    ## Data
    #Prices
    dum0=CSV.read(dirdata*"/pcouple.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    dum0=reshape(dum0,nold,T,K)
    ptemp=dum0

    ## Consumption
    dum0=CSV.read(dirdata*"/cvecouple.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    dum0=reshape(dum0,nold,T,K)
    cvetemp=dum0

    ## Interest rates
    dum0=CSV.read(dirdata*"/rvcouple.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    rvtemp=dum0.+1

    ################################################################################
    ## Main functions loading and initialization
    ################################################################################
    ## Moment: g(x,e).
    include(rootdir*"/cpufunctions/myfun.jl")
    ## Chain generation with CUDA.
    chainM[:,:,:]=zeros(n,dg,repn[2])
    include(rootdir*"/cudafunctions/cuda_chainfun.jl")
    ## Optimization with CUDA.
    numblocks = ceil(Int, n/100)
    include(rootdir*"/cudafunctions/cuda_fastoptim.jl")
    print("functions loaded!")

    @softscope for ri=1:npower
            #Fixing the seed.
            Random.seed!(123*ri)
            # Sampling prices from the original data.
            indsim=rand(1:2004,n)
            p=ptemp[indsim,:,:]
            rv=rvtemp[indsim,:]
            ## Discounted simulated prices.
            rho=zeros(n,T,K)
            for i=1:n
              for t=1:T
                rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
              end
            end
            rhoold=rho

            ###########################################################################################
            ## Data generation.
            # Generating consumption.
            cve=zeros(n,T,K)
            dlow=.8
            deltasim=rand(n).*(1-dlow).+dlow
            lambda=randexp(n)
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
            # This rescaling does not affect the Affriat inequalities and the moment conditions.
            # It is only needed to speed up the simulation and for stability of the numerical performance.
            cve=cve/1e5
            print("Data is ready!")
            ################################################################################
            ## Fixing random seed for the random number generator.
            Random.seed!(123*ri)
            ## Initializing gamma.
            gammav0=zeros(dg)
            #####################################################################################
            # Generating the first element of the chain.
            deltavec=[1.0]
            ndelta=length(deltavec)
            Delta=zeros(n)
            Deltatemp=zeros(n)
            W=ones(n,T,K)
            cvesim=zeros(n,T,K)
            vsim=zeros(n,T)
            optimval=ones(n,ndelta+1)*10000
            aiverify2=zeros(n,T,T)
            v=Variable(T, Positive())
            c=Variable(T,K,Positive())
            P=I+zeros(1,1)
            for dt=2:ndelta+1
                for id=1:n
                    Deltatemp[id]=deltavec[dt-1]
                    modvex=minimize(quadform(rho[id,1,:]'*(c[1,:]'-cve[id,1,:]),P)+quadform(rho[id,2,:]'*(c[2,:]'-cve[id,2,:]),P)+quadform(rho[id,3,:]'*(c[3,:]'-cve[id,3,:]),P)+quadform(rho[id,4,:]'*(c[4,:]'-cve[id,4,:]),P))
                    for t=1:T
                        for s=1:T
                            modvex.constraints+=v[t]-v[s]-Deltatemp[id]^(-(t-1))*rho[id,t,:]'*(c[t,:]'-c[s,:]')>=0
                        end
                    end
                    solve!(modvex,ECOSSolver(verbose=false))
                    optimval[id,dt]=modvex.optval
                    aiverify=zeros(n,T,T)
                    Delta[id]=Deltatemp[id]
                    for i=1:T
                        vsim[id,i]=v.value[i]
                        for j=1:K
                            cvesim[id,i,j]=c.value[i,j]
                        end
                    end

                    for t=1:T
                        for s=1:T
                            aiverify2[id,t,s]=vsim[id,t]-vsim[id,s]-Delta[id]^(-(t-1))*rho[id,t,:]'*(cvesim[id,t,:]-cvesim[id,s,:])
                        end
                    end
                end
                modvex=nothing
                GC.gc()
            end
            minimum(aiverify2)
            print("warm start ready!")

            Random.seed!(123*ri)
            println(ri)
            gchaincu!(theta0,gammav0,cve,rho,chainM,Delta,vsim,cvesim,W)
            print("chain ready!")
            ## Optimization step in CUDA
            Random.seed!(123*ri)
            indfast=rand(1:repn[2],nfast)
            indfast[1]=1
            chainMcu[:,:,:]=cu(chainM[:,:,indfast])
            numblocks = ceil(Int, n/100)
            include(rootdir*"/cudafunctions/cuda_fastoptim.jl")
            ###############################################################################
            ###############################################################################
            Random.seed!(123*ri)
            res = bboptimize(objMCcu2; SearchRange = (-10e300,10e300), NumDimensions = dg,MaxTime = 100.0, TraceMode=:silent)
            minr=best_fitness(res)
            TSMC=2*minr*n
            TSMC
            guessgamma=best_candidate(res)
            ###############################################################################
            ###############################################################################
            if (TSMC>=9.5)
                opt=NLopt.Opt(:LN_BOBYQA,dg)
                toluser=1e-6
                NLopt.lower_bounds!(opt,ones(dg).*-Inf)
                NLopt.upper_bounds!(opt,ones(dg).*Inf)
                NLopt.xtol_rel!(opt,toluser)
                NLopt.min_objective!(opt,objMCcu)
                (minf,minx,ret) = NLopt.optimize(opt, guessgamma)
                TSMC=2*minf*n
                TSMC
                solvegamma=minx
                print(ret)
                guessgamma=solvegamma
                ret
            end
            if (TSMC>=9.5)
                ##try 2
                (minf,minx,ret) = NLopt.optimize(opt, guessgamma)
                TSMC=2*minf*n
                TSMC
                solvegamma=minx
                guessgamma=solvegamma
                ret
            end
            if (TSMC>= 9.5)
                #try 3
                (minf,minx,ret) = NLopt.optimize(opt, guessgamma)
                TSMC=2*minf*n
                TSMC
                solvegamma=minx
                guessgamma=solvegamma
                ret
            end

            Resultspower[ri,2]=TSMC
            Resultspower[ri,1]=ri
            CSV.write(diroutput*"/B1_dgp1_chain_$repn.sample_$n.csv",Resultspower)
            GC.gc()
    end;
    Resultspower
end

try
    Results=powersimulations(chainM,chainMcu,theta0,n,repn,nfast)
catch
    @warn "Cuda needs a second run."
    Results=powersimulations(chainM,chainMcu,theta0,n,repn,nfast)
end
