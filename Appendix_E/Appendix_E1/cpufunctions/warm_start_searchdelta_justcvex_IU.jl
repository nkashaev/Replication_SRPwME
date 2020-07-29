##Guess quadratic program
using Convex
using ECOS

# Formulate and solve a simple quadratic problem:
#     min ||g|| st. Afriat inequalities for lambda=1

ndelta=1
lambdavec=ones(T,ndelta)

Lambda=zeros(n,T)
Lambdatemp=zeros(n,T)
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
        Lambdatemp[id,:]=lambdavec[:,dt-1]


        modvex=minimize(quadform(rho[id,1,:]'*(c[1,:]'-cve[id,1,:]),P)+quadform(rho[id,2,:]'*(c[2,:]'-cve[id,2,:]),P)+quadform(rho[id,3,:]'*(c[3,:]'-cve[id,3,:]),P)+quadform(rho[id,4,:]'*(c[4,:]'-cve[id,4,:]),P))
        for t=1:T
            for s=1:T
                modvex.constraints+=v[t]-v[s]-Lambdatemp[id,t]*rho[id,t,:]'*(c[t,:]'-c[s,:]')>=0
            end
        end

        solve!(modvex,ECOSSolver(verbose=false))

        optimval[id,dt]=modvex.optval



        aiverify=zeros(n,T,T)


        if (optimval[id,dt]<optimval[id,dt-1])
        Lambda[id,:]=Lambdatemp[id,:]
        for i=1:T
            vsim[id,i]=v.value[i]
            for j=1:K
                cvesim[id,i,j]=c.value[i,j]

            end
        end


        for t=1:T
            for s=1:T
                aiverify2[id,t,s]=vsim[id,t]-vsim[id,s]-Lambda[id,t]*rho[id,t,:]'*(cvesim[id,t,:]-cvesim[id,s,:])
            end
        end
    end
end
    modvex=nothing
    GC.gc()
end


minimum(aiverify2)
