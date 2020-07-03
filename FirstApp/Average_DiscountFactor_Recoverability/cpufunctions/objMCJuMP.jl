function objMCk(gamma1,gamma2,gamma3,gamma4)
    gamma0=[gamma1 gamma2 gamma3 gamma4]'
  geta=zeros(n,dg)
  gtry=zeros(n,dg)
  @inbounds geta[:,:]=chainM[:,:,1]
  dvecM=zeros(n,dg)
  #@simd for j=2:nfast
      @simd for j in indfast[2:end]
      gtry[:,:]=chainM[:,:,j]
      #logtrydens=gtry*gamma0-geta*gamma0+(-sum(gtry.^2,dims=2)+ sum(geta.^2,dims=2))[:,1,1]
      logtrydens=gtry*gamma0-geta*gamma0
      @inbounds dum=log.(rand(n)).<logtrydens
      @inbounds geta[dum,:]=gtry[dum,:]
      @inbounds dvecM[:,:]+=geta/(nfast-1)
    end
    dvec=sum(dvecM,dims=1)'/n


    numvar=zeros(T,T)
    @simd for i=1:n
        BLAS.syr!('U',1.0/n,dvecM[i,:],numvar)
    end
    var=numvar+numvar'- Diagonal(diag(numvar))-dvec*dvec'
    (Lambda,QM)=eigen(var)
    inddummy=Lambda.>0
    An=QM[:,inddummy]
    dvecdum2=An'*(dvec)
    vardum3=An'*var*An
    Omega2=inv(vardum3)
    Qn2=1/2*dvecdum2'*Omega2*dvecdum2

    return Qn2[1]
end
