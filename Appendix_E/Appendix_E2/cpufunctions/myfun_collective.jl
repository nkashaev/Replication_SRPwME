function myfun(;d=d,gamma=gamma,DeltaA=DeltaA,DeltaB=DeltaB,W=W,cve=cve,rho=rho)
    gvec=ones(n,T)
    @simd for j=1:T
      @inbounds gvec[:,j]=(sum((W[:,j,:]).*(rho[:,j,:]),dims=2))
    end
    gvec
end
