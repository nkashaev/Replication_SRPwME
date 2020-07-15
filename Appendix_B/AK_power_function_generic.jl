# ###Author: Victor H. Aguiar and Nail Kashaev
# ###email: vhaguiar@gmail.com
# ## Version: JULIA 1.1.0 (2019-01-21)
#
# ################################################################################
# ## Loading Packages and setting up procesors
###Author: Victor H. Aguiar and Nail Kashaev
###email: vhaguiar@gmail.com
## Version: JULIA 1.1.0 (2019-01-21)

################################################################################
## Loading Packages and setting up procesors
using LinearAlgebra
using Random
using MathProgBase
using Clp
using DataFrames
using JuMP
using Ipopt
using CSV
using NLopt
using BlackBoxOptim
#
using CuArrays
using CuArrays.CURAND
using CUDAnative
using CUDAdrv
#
using Convex
using SCS
using Clp
using ECOS

using SoftGlobalScope
###############################################################################
## Counterfactual
##
# data size
##seed
const T=4
const dg=4


###############################################################################
#Simulation sample size
const n=2000

## time length of the original data
T0=4
## number of goods
const K=17
#chain length
const repn=(0,10000)

chainM=zeros(n,dg,repn[2])
const nfast=10000
chainMcu=cu(chainM[:,:,1:nfast])

theta0=1.0

dgp="dgp1"

function powersimulations(chainM,chainMcu,theta0,n,repn,dgp)



    npower=length(gridvec)
    Resultspower=DataFrame(hcat(ones(npower),zeros(npower)))
    names!(Resultspower,Symbol.(["bshare","TSGMMcueMC"]))


    ## Setting-up directory
    tempdir1=@__DIR__
    repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
    appname="Appendix_F"
    rootdir=repdir*"/"*appname
    diroutput=repdir*"/Output_all/Appendix"
    dirdata=repdir*"/Data_all"

    ################################################################################
    ##
    # data size
    ##seed
    T=5
    dg=5


    ###############################################################################
    ## Data
    ## sample size
    #singles
    n=185

    ## time length of the original data
    T0=4
    ## number of goods
    K=17
    # repetitions for the simulation
    ## because the simulations are done using parallel Montecarlo we have 100*nprocs draws.
    repn=(0,10000)



    ###############################################################################
    ## Data
    ###############################################################################

    #Prices
    dum0=CSV.read(dirdata*"/p.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    dum0=reshape(dum0,n,T0,K)
    @eval  p=$dum0

    ## Consumption
    dum0=CSV.read(dirdata*"/cve.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    dum0=reshape(dum0,n,T0,K)
    @eval  cvetemp=$dum0

    ## Interest rates
    dum0=CSV.read(dirdata*"/rv.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    @eval rv=$dum0.+1

    ################################################################################
    ## Main functions loading and initialization

    ################################################################################
    ## moments
    ## Moment: my function
    include(rootdir*"/cpufunctions/myfun_counter.jl")
    ## chain generation with CUDA
    chainM[:,:,:]=zeros(n,dg,repn[2])
    include(rootdir*"/cudafunctions/cuda_chainfun.jl")
    ## optimization with CUDA
    numblocks = ceil(Int, n/100)
    # nfast=10000
    # Random.seed!(123)
    # indfast=rand(1:repn[2],nfast)
    # indfast[1]=1
    # GC.gc()

    chainMcu[:,:,:]=cu(chainM[:,:,indfast])
    include(rootdir*"/cudafunctions/cuda_fastoptim_counter.jl")
    print("functions loaded!")




    @softscope for ri=1:npower


            ## Discounted prices
            rho=zeros(n,T,K)
            for i=1:n
              for t=1:T0
                rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
              end
            end

            ###############################################################################
            ## Data Cleaning, Counterfactual prices

            ## Scaling up rho by kap and adjusting by  0.06 interest rate
            for i=1:n
                rho[i,T,:]=rho[i,T-1,:]/(1+0.06)
                rho[i,T,target]=rho[i,T,target]*kap
            end



            ## Set Consumption, we initialize the value of the latent consumption C^*_{T+1} to the value C^_{T0}
            cve=zeros(n,T,K)
            cve[:,1:T0,:]=cvetemp
            cve[:,T,:]=cvetemp[:,T0,:]
            cve

            print("load data ready!")


            ################################################################################
            ## Initializing

            ###########################################

            Random.seed!(123*ri)
            gammav0=zeros(dg)





            ## warmstart
            deltavec=theta0<1 ? [0 .5  1]*(1-theta0).+theta0 : [1]
            ndelta=length(deltavec)

            Delta=zeros(n)
            Deltatemp=zeros(n)
            W=ones(n,T,K)
            cvesim=zeros(n,T,K)
            vsim=zeros(n,T)
            optimval=ones(n,ndelta+1)*10000

            Kb=0
            aiverify2=zeros(n,T,T)
            v=Variable(T, Positive())
            c=Variable(T,K,Positive())
            P=I+zeros(1,1)



            for dt=2:ndelta+1
                for id=1:n
                    Deltatemp[id]=deltavec[dt-1]



                    #    return Delta, Alpha, W, vsim, cvesim


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
            print(bshare)
            gchaincu!(theta0,gammav0,cve,rho,chainM,bshare)
            print("chain ready!")


            ###########################################################################3
            ################################################################################################
            ## Optimization step in cuda
            chainMcu[:,:,:]=cu(chainM[:,:,indfast])
            include(rootdir*"/cudafunctions/cuda_fastoptim_counter.jl")


            ###############################################################################
            ###############################################################################
            Random.seed!(123*ri)
            res = bboptimize(objMCcu2c; SearchRange = (-10e300,10e300), NumDimensions = dg,MaxTime = 100.0, TraceMode=:silent)


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
            (minf,minx,ret) = NLopt.optimize(opt, guessgamma)
            TSMC=2*minf*n
            TSMC
            solvegamma=minx
            guessgamma=solvegamma
            ret
            ##try 2

            (minf,minx,ret) = NLopt.optimize(opt, guessgamma)
            TSMC=2*minf*n
            TSMC
            solvegamma=minx
            guessgamma=solvegamma
            ret

            #try 3

            (minf,minx,ret) = NLopt.optimize(opt, guessgamma)
            TSMC=2*minf*n
            TSMC
            solvegamma=minx
            guessgamma=solvegamma
            ret


        ########
            Resultspower[ri,2]=TSMC
            Resultspower[ri,1]=bshare
            CSV.write(diroutput*"/counter.good_$targetgood._price_$target._multiplier_$kap._cuda_start.$startit.end.$endit.theta0.$theta0.csv",Resultspower)
            GC.gc()


    end;
    Resultspower
end

Results=counterbounds(chainM,chainMcu,indfast,theta0,targetgood,target)
