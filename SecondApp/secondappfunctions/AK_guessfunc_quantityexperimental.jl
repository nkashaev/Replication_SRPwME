@everywhere function guessfun(;d=d::Float64,gamma=gamma::Float64,cve=cve::Float64,rho=rho::Float64)
    nobs=T
    ngoods=K
    afriatpar=1
    seed=rand(1:1000000000)
    maxit=100000
    R"set.seed($seed)"
    for indz=1:n
      ptest=rho[indz,:,:]
      wtest=mrep[indz,:,:]
      R"res2=revealedPrefsmod::simGarpPriceWealth(nobs=$nobs,ngoods=$ngoods,afriat.par=$afriatpar,maxit=$maxit,qmin=0,qmax=1,pmin=0,pmax=1,p=$ptest,w=$wtest)"
      R"res2x=res2$x"
      @rget res2x
      ntest=size(res2x)[1]
      maxit2=1
      while (ntest<T | maxit2<=100000)
        seed=rand(1:1000000000)
        R"set.seed($seed)"
        R"res2=revealedPrefsmod::simGarpPriceWealth(nobs=$nobs,ngoods=$ngoods,afriat.par=$afriatpar,maxit=$maxit,qmin=0,qmax=1,pmin=0,pmax=1,p=$ptest,w=$wtest)"
        R"res2x=res2$x"
        maxit2=maxit2+1
        @rget res2x
        ntest=size(res2x)[1]
      end
      cvesim[indz,:,:]=res2x
    end
    hcat(reshape(cvesim,n,T*K))
end
