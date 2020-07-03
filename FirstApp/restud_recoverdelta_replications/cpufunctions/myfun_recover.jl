@everywhere function myfun(;d=d,gamma=gamma,Delta=Delta,W=W,cve=cve,rho=rho)

    gvec=ones(n,T)


    @simd for j=1:(T-1)
      @inbounds gvec[:,j]=(sum((W[:,j,:]).*(rho[:,j,:]),dims=2))
    end

    gvec[:,T-1]=(cve[:,T-1,targetgood]-W[:,T-1,targetgood]).*rho[:,T-1,targetgood]./sum(rho[:,T-1,:].*(cve[:,T-1,:].-W[:,T-1,:]),dims=2)[:,1]- bshare*ones(n)
    gvec
end
