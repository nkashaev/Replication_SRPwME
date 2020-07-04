@everywhere function myfun(;d=d,gamma=gamma,Delta=Delta,W=W,cve=cve,rho=rho)

    gvec=ones(n,dg)


    @simd for j=1:(T)
      @inbounds gvec[:,j]=(sum((W[:,j,:]).*(rho[:,j,:]),dims=2))
    end

    gvec[:,dg]=(cve[:,T,target]-W[:,T,target]).*rho[:,T,target]./sum(rho[:,T,:].*(cve[:,T,:].-W[:,T,:]),dims=2)[:,1]- bshare*ones(n)
    gvec
end
