using Convex
using ECOS
# deltavec is a vector of candidates fixed delta values
# Here we fix it to 1, to keep the code comparable across different values of tehta0
deltavec=[1]
# An example of another deltavec=[0 0.5 1].*(1-theta0) .+ theta0.
ndelta=length(deltavec)
# Initialize Delta vector.
Delta=zeros(n)
# Initialize temporary  best candidate of Delta vector.
Deltatemp=zeros(n)
# Initialize W^c=W (consumption measurement error).
W=ones(n,T,K)
# Initialize simulated consumption value C^*=cvesim.
cvesim=zeros(n,T,K)
# Initialize simulated Afriat numbers v=vsim.
vsim=zeros(n,T)
# Initialize optimal value to a high number
optimval=ones(n,ndelta+1)*10000
# Initialize matrix to verify Afriat inequalities of the simulated data.
aiverify2=zeros(n,T,T)
# Convex model, variable definition.
v=Variable(T, Positive())
c=Variable(T,K,Positive())
# Identity Matrix in Convex.
P=I+zeros(1,1)


# Loop to check all entries of deltavec per individual and save in Delta the value corresponding to the lowest objective value
for dt=2:ndelta+1
    for id=1:n
        Deltatemp[id]=deltavec[dt-1]
        # Convex objective value, see Appendix C.
        modvex=minimize(quadform(rho[id,1,:]'*(c[1,:]'-cve[id,1,:]),P)+quadform(rho[id,2,:]'*(c[2,:]'-cve[id,2,:]),P)+quadform(rho[id,3,:]'*(c[3,:]'-cve[id,3,:]),P)+quadform(rho[id,4,:]'*(c[4,:]'-cve[id,4,:]),P))
        # Define the Afriat's constraints.
        for t=1:T
            for s=1:T
                modvex.constraints+=v[t]-v[s]-Deltatemp[id]^(-(t-1))*rho[id,t,:]'*(c[t,:]'-c[s,:]')>=0
            end
        end
        # Solve the model using ECOS.
        solve!(modvex,ECOSSolver(verbose=false))
        # Save objective value.
        optimval[id,dt]=modvex.optval
        # Re-initialize the verification matrix.
        aiverify=zeros(n,T,T)

        # Keep the lowest objective value, associated Delta
        if (optimval[id,dt]<optimval[id,dt-1])
            Delta[id]=Deltatemp[id]
            for i=1:T
                vsim[id,i]=v.value[i]
                for j=1:K
                    cvesim[id,i,j]=c.value[i,j]
                end
            end
            # Verify Afriat's inequalities with the solution.
            for t=1:T
                for s=1:T
                    aiverify2[id,t,s]=vsim[id,t]-vsim[id,s]-Delta[id]^(-(t-1))*rho[id,t,:]'*(cvesim[id,t,:]-cvesim[id,s,:])
                end
            end
        end
    end
    #Memory cleaning.
    modvex=nothing
    GC.gc()
end

minimum(aiverify2)
