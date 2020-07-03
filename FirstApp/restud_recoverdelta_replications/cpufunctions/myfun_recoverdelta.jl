function myfun(;d=d,gamma=gamma,Delta=Delta,W=W,cve=cve,rho=rho)

    gvec=ones(n,dg)


    @simd for j=1:(T)
      @inbounds gvec[:,j]=(sum((W[:,j,:]).*(rho[:,j,:]),dims=2))
    end

    gvec[:,dg]=Delta - avgdelta*ones(n)
    gvec
end
