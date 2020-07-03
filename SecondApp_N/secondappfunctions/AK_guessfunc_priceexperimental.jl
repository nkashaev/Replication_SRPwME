#Guessfun: gives the initial draw of the Montecarlo step, must be a simulations consistent with the null.
## Here it invokes the revealedPrefsmod function simGarpQuantWealth, that will generate a draw of p^* that satisfies GARP and in on the budget line.
## The function allows to set an afriatpar that corresponds to the cost efficiency index. We set it to 1.
#maxit is the max. number of iterations allowed by the sampler before it restarts.
#R has to get a random seed.
#Do not pay attention to the name of the files cvesim since it does not matter, in this case it is filled by prices
@everywhere function guessfun(;d=d::Float64,gamma=gamma::Float64,cve=cve::Float64,rho=rho::Float64)
    nobs=T
    ngoods=K
    afriatpar=1
    seed=rand(1:1000000000)
    maxit=100000
    R"set.seed($seed)"
    for indz=1:n
      qtest=cve[indz,:,:]
      wtest=mrep[indz,:,:]
      R"res2=revealedPrefsmod::simGarpQuantWealth(nobs=$nobs,ngoods=$ngoods,afriat.par=$afriatpar,maxit=$maxit,qmin=0,qmax=1,pmin=0,pmax=1,q=$qtest,w=$wtest)"

      R"res2x=res2$p"

      @rget res2x
      ntest=size(res2x)[1]
      maxit2=1
      while (ntest<T | maxit2<=100000)
        seed=rand(1:1000000000)
        R"set.seed($seed)"
        R"res2=revealedPrefsmod::simGarpQuantWealth(nobs=$nobs,ngoods=$ngoods,afriat.par=$afriatpar,maxit=$maxit,qmin=0,qmax=1,pmin=0,pmax=1,q=$qtest,w=$wtest)"

        R"res2x=res2$p"
        maxit2=maxit2+1
        @rget res2x
        ntest=size(res2x)[1]
      end

      cvesim[indz,:,:]=res2x

    end
    hcat(reshape(cvesim,n,T*K))
end
