##Guess quadratic program
using Convex
using SCS
using JuMP
using Ipopt
#using CPLEX
using Clp
using ECOS

###############################################################################
#theta0=d

#end
# """
# Formulate and solve a simple LP:
#     max ||g|| st. Afriat inequalities for random d
# """
#function guessfun(;d=d,gamma=gamma,cve=cve,rho=rho)
#theta0=d

#deltavec=theta0<1 ? [0 .1 .2 .3 .4 .5 .6 .7 .8 .9 1]*(1-theta0).+theta0 : [1]
#deltavec=theta0<1 ? [0 .1 .15 .2 .25 .3 .35 .4 .45 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95 1]*(1-theta0).+theta0 : [1]
## Power
theta0=1.0
#deltavec=theta0<1 ? [0 .5  1]*(1-theta0).+theta0 : [1]
ndelta=10
lambdavec=ones(T,ndelta)
lambdavec[2:T,2:end]=randexp(T-1,ndelta-1)
#deltavec=theta0<1 ? [0  1]*(1-theta0).+theta0 : [1]
#ndelta=length(lambdavec[1,:])

Lambda=zeros(n,T)
Lambdatemp=zeros(n,T)
Alpha=zeros(n)
W=ones(n,T,K)
cvesim=zeros(n,T,K)
vsim=zeros(n,T)
optimval=ones(n,ndelta+1)*10000

Kb=0
Alpha[:]=(rand(n).*(1-.1).+.1).^(-Kb)
aiverify2=zeros(n,T,T)
#for id=1:n
v=Variable(T, Positive())
c=Variable(T,K,Positive())
P=I+zeros(1,1)



for dt=2:ndelta+1
    for id=1:n
        ##dws=rand()*(1-theta0)+theta0
        Lambdatemp[id,:]=lambdavec[:,dt-1]



        #    return Delta, Alpha, W, vsim, cvesim


        modvex=minimize(quadform(rho[id,1,:]'*(c[1,:]'-cve[id,1,:]),P)+quadform(rho[id,2,:]'*(c[2,:]'-cve[id,2,:]),P)+quadform(rho[id,3,:]'*(c[3,:]'-cve[id,3,:]),P)+quadform(rho[id,4,:]'*(c[4,:]'-cve[id,4,:]),P))
        #modvex=minimize(norm(rho[id,1,:]'*(c[1,:]'-cve[id,1,:]),2)+norm(rho[id,2,:]'*(c[2,:]'-cve[id,2,:]),2)+norm(rho[id,3,:]'*(c[3,:]'-cve[id,3,:]),2)+norm(rho[id,4,:]'*(c[4,:]'-cve[id,4,:]),2))
        for t=1:T
            for s=1:T
                modvex.constraints+=v[t]-v[s]-Lambdatemp[id,t]*rho[id,t,:]'*(c[t,:]'-c[s,:]')>=0
            end
        end

        #solve!(modvex,SCSSolver(max_iters=500000))
        #solve!(modvex,ECOSSolver())
        #solve!(modvex,CplexSolver())
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


# indreoptim=collect(findall(x->x<0,aiverify2))
# indreoptim2=[]
# for j=1:size(indreoptim)[1]
#     push!(indreoptim2,indreoptim[j][1])
# end
# indreoptim3=unique(indreoptim2)
# indreoptim3
#
#
# for id in indreoptim3
#
#         modelws = Model(with_optimizer(Ipopt.Optimizer,print_level=0))
#         @variable(modelws, v[1:T] >= 0)
#         @variable(modelws, 0 <= c[1:T,1:K] <= 1e10)
#         @constraint(modelws,aicons[t in 1:T,s in 1:T],v[t]-v[s]-Delta[id]^(-(t-1))*sum(rho[id,t,l].*(c[t,l]-c[s,l]) for l in 1:K)>=0)
#
#         @objective(modelws, Max, -sum((sum(rho[id,t,l].*(c[t,l]-cve[id,t,l]) for l in 1:K))^2 for t in 1:T))
#
#         JuMP.optimize!(modelws)
#         #optimval[id,dt] = JuMP.objective_value(modelws)
#
#         for i=1:T
#             vsim[id,i]=JuMP.value(v[i])
#             for j=1:K
#                 cvesim[id,i,j]=JuMP.value(c[i,j])
#             end
#         end
#
#
#
#
#
#
#
#         for t=1:T
#             for s=1:T
#                 aiverify2[id,t,s]=vsim[id,t]-vsim[id,s]-Delta[id]^(-(t-1))*rho[id,t,:]'*(cvesim[id,t,:]-cvesim[id,s,:])
#             end
#         end
#         modelws=nothing
#         GC.gc()
# end
#
#
# indreoptim=collect(findall(x->x<0,aiverify2))
# indreoptim2=[]
# for j=1:size(indreoptim)[1]
#     push!(indreoptim2,indreoptim[j][1])
# end
# indreoptim4=unique(indreoptim2)
# indreoptim4
#
# #############################################################################
# for id in indreoptim4
#     #dws=rand()*(1-theta0)+theta0
#     #Delta[id]=dws
#     modelpar = Model(with_optimizer(Ipopt.Optimizer,print_level=0))
#     #@variable(modelpar, v[1:T] >= 0)
#     @variable(modelpar, 0.5 <= sigma[1:K] <= 1)
#     @variable(modelpar, lambda >=0.1 )
#
#     @NLobjective(modelpar, Max, -sum((sum(rho[id,t,l]*((lambda/(Delta[id]^(t-1))*rho[id,t,l])^(-1/sigma[l])-cve[id,t,l]) for l in 1:K))^2 for t in 1:T))
#     JuMP.optimize!(modelpar)
#     obj_value = JuMP.objective_value(modelpar)
#
#
#     for i=1:T
#         for j=1:K
#             cvesim[id,i,j]=(JuMP.value(lambda)/(Delta[id]^(i-1))*rho[id,i,j])^(-1/JuMP.value(sigma[j]))
#         end
#     end
#
#     vsim[id,:]=zeros(T)
#     for j=1:K
#         vsim[id,:]+=cvesim[id,:,j].^(1-JuMP.value(sigma[j]))./(1-JuMP.value(sigma[j]))./JuMP.value(lambda)
#     end
#     for t=1:T
#         for s=1:T
#             aiverify2[id,t,s]=vsim[id,t]-vsim[id,s]-Delta[id]^(-(t-1))*rho[id,t,:]'*(cvesim[id,t,:]-cvesim[id,s,:])
#         end
#     end
#     modelws=nothing
#     GC.gc()
# end
#
# indfail=findall(isequal(NaN),aiverify2)
# indfail2=[]
# for j=1:size(indfail)[1]
#     push!(indfail2,indfail[j][1])
# end
# indfail5=unique(indfail2)
#
#
# indreoptim=collect(findall(x->x<0,aiverify2))
# indreoptim2=[]
# for j=1:size(indreoptim)[1]
#     push!(indreoptim2,indreoptim[j][1])
# end
# indreoptim5=unique(indreoptim2)
# indreoptim5
#
#
# W[:,:,:]=cve[:,:,:]-cvesim[:,:,:]
#
# minimum(aiverify2)
L
