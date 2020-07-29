function myfun(;d=d,gamma=gamma,Delta=Delta,W=W,cve=cve,rho=rho,bshare=bshare)
    gvec=ones(n,T)
    @simd for j=1:(T-1)
      @inbounds gvec[:,j]=(sum((W[:,j,:]).*(rho[:,j,:]),dims=2))
    end
    gvec[:,T]=(cve[:,T,targetgood]-W[:,T,targetgood]).*rho[:,T,targetgood]./sum(rho[:,T,:].*(cve[:,T,:].-W[:,T,:]),dims=2)[:,1]- bshare*ones(n)
    gvec
end
