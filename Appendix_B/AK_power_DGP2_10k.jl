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
## DGP1
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



function powersimulations(chainM,chainMcu,theta0,n,repn,nfast)



    npower=1000
    Resultspower=DataFrame(hcat(ones(npower),zeros(npower)))
    names!(Resultspower,Symbol.(["iter","TSGMMcueMC"]))


    ## Setting-up directory
    tempdir1=@__DIR__
    repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
    appname="Appendix_B"
    rootdir=repdir*"/"*appname
    diroutput=repdir*"/Output_all/Appendix"
    dirdata=repdir*"/Data_all"

    ## data size
    # Sample size
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
    ## moments
    ## Moment: my function
    include(rootdir*"/cpufunctions/myfun.jl")
    ## chain generation with CUDA
    chainM[:,:,:]=zeros(n,dg,repn[2])
    include(rootdir*"/cudafunctions/cuda_chainfun.jl")
    ## optimization with CUDA
    numblocks = ceil(Int, n/100)
    include(rootdir*"/cudafunctions/cuda_fastoptim.jl")
    print("functions loaded!")




    @softscope for ri=1:npower
            Random.seed!(123*ri)
            indsim=rand(1:2004,n)
            p=ptemp[indsim,:,:]
            rv=rvtemp[indsim,:]




            ## Discounted prices
            rho=zeros(n,T,K)
            for i=1:n
              for t=1:T0
                rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
              end
            end
            rhoold=rho

            ###########################################################################################
            ## Data generation DGP2
            cve=zeros(n,T,K)
            dlow=1.0
            deltasim=rand(n).*(1-dlow).+dlow
            lambda=randexp(n)/1
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

            #cve=cve/1e5


            print("load data ready!")


            ################################################################################
            ## Initializing

            ###########################################

            Random.seed!(123)
            gammav0=zeros(dg)




            #####################################################################################
            ## warmstart
            deltavec=[.8]
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
            W[:,:,:]=cve[:,:,:]-cvesim[:,:,:]

            minimum(aiverify2)
            print("warm start ready!")

            Random.seed!(123*ri)
            println(ri)
            gchaincu!(theta0,gammav0,cve,rho,chainM,Delta,vsim,cvesim,W)
            print("chain ready!")


            ###########################################################################3
            ################################################################################################
            ## Optimization step in cuda
            Random.seed!(123*ri)
            indfast=rand(1:repn[2],nfast)
            indfast[1]=1
            chainMcu[:,:,:]=cu(chainM[:,:,indfast])
            numblocks = ceil(Int, n/100)
            include(rootdir*"/cudafunctions/cuda_fastoptim.jl")


            ###############################################################################
            ###############################################################################
            Random.seed!(123)
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

            #try 3
            if (TSMC>= 9.5)
                (minf,minx,ret) = NLopt.optimize(opt, guessgamma)
                TSMC=2*minf*n
                TSMC
                solvegamma=minx
                guessgamma=solvegamma
                ret
            end
            #try 4
            if (TSMC>= 9.5)
                (minf,minx,ret) = NLopt.optimize(opt, guessgamma)
                TSMC=2*minf*n
                TSMC
                solvegamma=minx
                guessgamma=solvegamma
                ret
            end


        ########
            Resultspower[ri,2]=TSMC
            Resultspower[ri,1]=ri
            CSV.write(diroutput*"/B_power_dgp2_chain_$repn.sample_$n.theta0.$theta0.csv",Resultspower)
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
