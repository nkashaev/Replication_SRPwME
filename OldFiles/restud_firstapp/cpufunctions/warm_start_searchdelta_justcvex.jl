##Guess quadratic program
using Convex
using SCS
using ECOS

## Power
#deltavec=theta0<1 ? [0 .1 .15 .2 .25 .3 .35 .4 .45 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95 1]*(1-theta0).+theta0 : [1]
#deltavec=theta0<1 ? [0 .1 .2 .3 .4 .5 .6 .7 .8 .9 1]*(1-theta0).+theta0 : [1]
#deltavec=theta0<1 ? [0 .5  1]*(1-theta0).+theta0 : [1]
deltavec=[1]
ndelta=length(deltavec)

Delta=zeros(n)
Deltatemp=zeros(n)
Alpha=zeros(n)
W=ones(n,T,K)
cvesim=zeros(n,T,K)
vsim=zeros(n,T)
optimval=ones(n,ndelta+1)*10000

Kb=0
Alpha[:]=(rand(n).*(1-.1).+.1).^(-Kb)
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


        if (optimval[id,dt]<optimval[id,dt-1])
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
end
    modvex=nothing
    GC.gc()
end

minimum(aiverify2)
