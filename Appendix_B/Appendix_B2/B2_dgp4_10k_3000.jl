## The code below is almost the same as the code for DGP4 (B1_dgp4_10k_2000.jl)
# The only difference is that n=3000
using LinearAlgebra
using Random
using MathProgBase
using DataFrames
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
using ECOS

using SoftGlobalScope

#
const T=4
const dg=4
const n=3000
const K=17
const repn=(0,10000)
chainM=zeros(n,dg,repn[2])
const nfast=10000
chainMcu=cu(chainM[:,:,1:nfast])

theta0=.1

function powersimulations(chainM,chainMcu,theta0,n,repn,nfast)
    npower=1000
    Resultspower=DataFrame(hcat(ones(npower),zeros(npower)))
    names!(Resultspower,Symbol.(["iter","TSGMMcueMC"]))

    tempdir1=@__DIR__
    repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
    appname="Appendix_B"
    rootdir=repdir*"/"*appname
    diroutput=repdir*"/Output_all/Appendix"
    dirdata=repdir*"/Data_all"

    nold=2004
    T=4
    K=17
    dg=T

    dum0=CSV.read(dirdata*"/pcouple.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    dum0=reshape(dum0,nold,T,K)
    ptemp=dum0

    dum0=CSV.read(dirdata*"/cvecouple.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    dum0=reshape(dum0,nold,T,K)
    cvetemp=dum0

    dum0=CSV.read(dirdata*"/rvcouple.csv",datarow=2,allowmissing=:none)
    dum0=convert(Matrix,dum0[:,:])
    rvtemp=dum0.+1

    include(rootdir*"/cpufunctions/myfun.jl")
    chainM[:,:,:]=zeros(n,dg,repn[2])
    include(rootdir*"/cudafunctions/cuda_chainfun.jl")
    numblocks = ceil(Int, n/100)
    include(rootdir*"/cudafunctions/cuda_fastoptim.jl")
    print("functions loaded!")

    @softscope for ri=1:npower
            Random.seed!(123*ri)
            indsim=rand(1:2004,n)
            p=ptemp[indsim,:,:]
            rv=rvtemp[indsim,:]
            rho=zeros(n,T,K)
            for i=1:n
              for t=1:T
                rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
              end
            end
            rhoold=rho

            cve=zeros(n,T,K)
            dlowa=.1
            deltasima=rand(n).*(1-dlowa).+dlowa
            dlowb=.99
            dhighb=1.0
            deltasimb=rand(n).*(dhighb-dlowb).+dlowb
            lambda=randexp(n)/1
            lambdab=randexp(n)/1
            mulo=1/3
            muhi=2/3
            mu=rand(n,T,K).*(muhi-mulo).+mulo
            su=100
            sl=1/15
            sigma=rand(n,K)*(su-sl) .+ sl
            sigmab=rand(n,K)*(su-sl) .+ sl
            adum=0.97
            bdum=1.03
            epsilon=adum .+ rand(n,T,K)*(bdum-adum)
            @simd for i=1:n
               for t=1:T
                 for k=1:K
                   rhoadum=mu[i,t,k]*rhoold[i,t,k]
                   rhobdum=rhoold[i,t,k]-rhoadum
                   cvea=((lambda[i]/deltasima[i]^(t-1))*rhoadum).^(-1/sigma[i,k])
                   cveb=((lambdab[i]/deltasimb[i]^(t-1))*rhobdum).^(-1/sigmab[i,k])
                   cve[i,t,k]=  (cvea+cveb)*epsilon[i,t,k]
                   rho[i,t,k]=rhoadum+rhobdum
                 end
               end
             end
            cve=cve/1e7
            print("load data ready!")

            Random.seed!(123*ri)
            gammav0=zeros(dg)

            deltavec=theta0<1 ? [0 .5  1]*(1-theta0).+theta0 : [1]
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

            Random.seed!(123*ri)
            indfast=rand(1:repn[2],nfast)
            indfast[1]=1
            chainMcu[:,:,:]=cu(chainM[:,:,indfast])
            numblocks = ceil(Int, n/100)
            include(rootdir*"/cudafunctions/cuda_fastoptim.jl")

            Random.seed!(123*ri)
            res = bboptimize(objMCcu2; SearchRange = (-10e300,10e300), NumDimensions = dg,MaxTime = 100.0, TraceMode=:silent)
            minr=best_fitness(res)
            TSMC=2*minr*n
            TSMC
            guessgamma=best_candidate(res)

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
                (minf,minx,ret) = NLopt.optimize(opt, guessgamma)
                TSMC=2*minf*n
                TSMC
                solvegamma=minx
                guessgamma=solvegamma
                ret
            end
            if (TSMC>= 9.5)
                (minf,minx,ret) = NLopt.optimize(opt, guessgamma)
                TSMC=2*minf*n
                TSMC
                solvegamma=minx
                guessgamma=solvegamma
                ret
            end

            Resultspower[ri,2]=TSMC
            Resultspower[ri,1]=ri
            CSV.write(diroutput*"/B2_dgp4_chain_$repn.sample_$n.csv",Resultspower)
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
