function myfun(;d=d,gamma=gamma,Lambda=Lambda,Delta=Delta,W=W,cve=cve,rho=rho)
    gvec=ones(n,dg)
    @simd for j=1:(T)
      @inbounds gvec[:,j]=(sum((W[:,j,:]).*(rho[:,j,:]),dims=2))
    end
    gvec[:,T+1]=Delta[:].^(1).*Lambda[:,2] .-1
    gvec[:,T+2]=Delta[:].^(2).*Lambda[:,3] .-1
    gvec[:,T+3]=Delta[:].^(3).*Lambda[:,4] .-1
    gvec
end
