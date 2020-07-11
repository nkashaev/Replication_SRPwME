function warmstart_nail(rho,cve,d_ini)
      (n,T,K)=size(rho)
      Delta=d_ini*ones(n)
      W=ones(n,T,K)
      cvesim=zeros(n,T,K)
      vsim=zeros(n,T)
      optimval=ones(n)*10000

      aiverify2=zeros(n,T,T)
      v=Variable(T, Positive())
      c=Variable(T,K,Positive())
      P=I+zeros(1,1)

      for id=1:n
        d=Delta[id]
        modvex=minimize(quadform(rho[id,1,:]'*(c[1,:]'-cve[id,1,:]),P)+quadform(rho[id,2,:]'*(c[2,:]'-cve[id,2,:]),P)+quadform(rho[id,3,:]'*(c[3,:]'-cve[id,3,:]),P)+quadform(rho[id,4,:]'*(c[4,:]'-cve[id,4,:]),P))
        for t=1:T, s=1:T
             modvex.constraints+=v[t]-v[s]-d^(-(t-1))*rho[id,t,:]'*(c[t,:]'-c[s,:]')>=0
        end
        solve!(modvex,ECOSSolver(verbose=false))
        optimval[id]=modvex.optval
        for i=1:T
            vsim[id,i]=v.value[i]
            for j=1:K
                cvesim[id,i,j]=c.value[i,j]
            end
        end
      end
      modvex=nothing
      GC.gc()

      W[:,:,:]=cve[:,:,:]-cvesim[:,:,:]

      #minimum(aiverify2)
      return cvesim, vsim, Delta, W
end
