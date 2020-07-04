@everywhere function myfun(;d=d,gamma=gamma,Delta=Delta,Lambda=Lambda,W=W,cve=cve,rho=rho)

    gvec=ones(n,dg)


    @simd for j=1:(T)
      @inbounds gvec[:,j]=(sum((W[:,j,:]).*(rho[:,j,:]),dims=2))
    end

    gvec[:,T+1]=Lambda[:,2] .-1
    gvec[:,T+2]=Lambda[:,3] .-1
    gvec[:,T+3]=Lambda[:,4] .-1
    gvec[:,T+4]=(Lambda[:,4]-Lambda[:,3]).*Lambda[:,3]
    gvec[:,T+5]=(Lambda[:,3]-Lambda[:,2]).*Lambda[:,2]
    gvec
end
