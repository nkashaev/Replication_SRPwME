####################################################################################
@everywhere function gchain4(;d=d,gamma=gamma,cve=cve,rho=rho,chainM=chainM)
    # Delta=zeros(n)
    # Alpha=zeros(n)
    # W=ones(n,T,K)
    # cvesim=zeros(n,T,K)
    # vsim=zeros(n,T)

    Deltac=zeros(n)
    Wc=ones(n,T,K)
    cvesimc=zeros(n,T,K)
    vsimc=zeros(n,T)

    a=zeros(n,dg)

    #Delta[:],Alpha[:],W[:,:,:],vsim[:,:],cvesim[:,:,:]=guessfun(d=d,gamma=gamma,cve=cve,rho=rho);
    r=-repn[1]+1
    while r<=repn[2]
      Deltac[:],Wc[:,:,:],vsimc[:,:],cvesimc[:,:,:]=jumpfun2(d=d,gamma=gamma,Delta=Delta,cvesim=cvesim,vsim=vsim,cve=cve,rho=rho);
      logtrydens=(-sum(sum(rho.*Wc,dims=3).^2,dims=2)+ sum(sum(rho.*W,dims=3).^2,dims=2))[:,1,1]
      dum=log.(rand(n)).<logtrydens

      #dum=zeros(n).<ones(n)
      @inbounds cvesim[dum,:,:]=cvesimc[dum,:,:]
      @inbounds W[dum,:,:]=Wc[dum,:,:]
      @inbounds vsim[dum,:]=vsimc[dum,:]
      @inbounds Delta[dum,:]=Deltac[dum,:]
      if r>0
        chainM[:,:,r]=myfun(d=d,gamma=gamma,Delta=Delta,W=W,cve=cve,rho=rho)

      end
      r=r+1
    end

end
