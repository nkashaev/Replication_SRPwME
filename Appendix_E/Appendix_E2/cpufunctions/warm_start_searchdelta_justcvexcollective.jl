##Guess quadratic program
using Convex
using SCS
using JuMP
using Ipopt
using Clp
using ECOS


deltavecA=darandsim
deltavecB=dbrandsim

ndelta=1

DeltaA=zeros(n)
DeltatempA=zeros(n)
DeltaB=zeros(n)
DeltatempB=zeros(n)
W=ones(n,T,K)
cvesim=zeros(n,T,K)
vsimA=zeros(n,T)
vsimB=zeros(n,T)
optimval=ones(n,ndelta+1)*10000

Kb=0
aiverify2=zeros(n,T,T)
vA=Variable(T, Positive())
vB=Variable(T, Positive())
c=Variable(T,K,Positive())
P=I+zeros(1,1)



for id=1:n
        DeltatempA[id]=deltavecA[id]
        DeltatempB[id]=deltavecB[id]


        #    return Delta, Alpha, W, vsim, cvesim


        modvex=minimize(quadform(rho[id,1,:]'*(c[1,:]'-cve[id,1,:]),P)+quadform(rho[id,2,:]'*(c[2,:]'-cve[id,2,:]),P)+quadform(rho[id,3,:]'*(c[3,:]'-cve[id,3,:]),P)+quadform(rho[id,4,:]'*(c[4,:]'-cve[id,4,:]),P))
        for t=1:T
            for s=1:T
                modvex.constraints+=DeltatempA[id]^(-(t-1))*(vA[t]-vA[s])+DeltatempB[id]^(-(t-1))*(vB[t]-vB[s])-rho[id,t,:]'*(c[t,:]'-c[s,:]')>=0
            end
        end

        solve!(modvex,ECOSSolver(verbose=false))

        optimval[id,1]=modvex.optval



        aiverify=zeros(n,T,T)


        DeltaA[id]=DeltatempA[id]
        DeltaB[id]=DeltatempB[id]

        for i=1:T
            vsimA[id,i]=vA.value[i]
            vsimB[id,i]=vB.value[i]
            for j=1:K
                cvesim[id,i,j]=c.value[i,j]

            end
        end








        for t=1:T
            for s=1:T
                aiverify2[id,t,s]=DeltaA[id]^(-(t-1))*(vsimA[id,t]-vsimA[id,s])+DeltaB[id]^(-(t-1))*(vsimB[id,t]-vsimB[id,s])-rho[id,t,:]'*(cvesim[id,t,:]-cvesim[id,s,:])
            end
        end
    #end
end
modvex=nothing
GC.gc()

minimum(aiverify2)
