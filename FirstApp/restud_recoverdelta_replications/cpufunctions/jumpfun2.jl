##################################################################################
@everywhere function jumpfun2(;d=d,gamma=gamma,Delta=Delta,vsim=vsim,cvesim=cvesim,cve=cve,rho=rho)

  Deltac=zeros(n)
  Wc=ones(n,T,K)
  cvesimc=zeros(n,T,K)
  vsimc=zeros(n,T)
  for isim=1:n
    dmax,dmin=new_delta(d=d,vsim=vsim[isim,:],cvesim=cvesim[isim,:,:],rho=rho[isim,:,:])
    Deltac[isim]=rand()*(dmax-dmin).+dmin
    vsimc[isim,:], cvesimc[isim,:,:]=new_VC_long(hcat(vsim[isim,:],cvesim[isim,:,:]),([Deltac[isim].^0; Deltac[isim].^(-1); Deltac[isim].^(-2); Deltac[isim].^(-3)].*rho[isim,:,:]))
  end
  Wc[:,:,:]=cve[:,:,:]-cvesimc[:,:,:]
  return Deltac, Wc, vsimc, cvesimc
end
