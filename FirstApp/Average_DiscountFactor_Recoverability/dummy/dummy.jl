d=theta0
dcu=cu(d)
Deltac=zeros(n)
Wc=ones(n,T,K)
cvesimc=zeros(n,T,K)
vsimc=zeros(n,T)
  ##cuda
Deltacu=cu(Delta)
vsimcu=cu(vsim)
cvesimcu=cu(cvesim)
  ##candidates
Deltaccu=cu(Deltac)
vsimccu=cu(vsimc)
cvesimccu=cu(cvesimc)
  ## auxiliaries
cvecu=cu(cve)
rhocu=cu(rho)
VCu=cu(zeros(n,T,K+1))



    #unif1=curand(n).*(.999-.001).+.001
unif1=curand(n)
v=curandn(n,T,(K+1))
dVC=v./norm(v)
    #unif2=curand(n).*(.999-.001).+.001
unif2=curand(n)
numblocks = ceil(Int, n/167)
VC=VCu
@cuda threads=167 blocks=numblocks jumpfuncu!(dcu,Deltacu,vsimcu,cvesimcu,rhocu,Deltaccu,vsimccu,cvesimccu,unif1,unif2,VC,dVC)
#Array(Deltac), Array(cve-cvesimc), Array(vsimc), Array(cvesimc)

function new_deltacu!(d,vsim,cvesim,rho,Deltac,isim,unif1)
    dmin=d
    dmax=1

    for t=2:5

        for s=1:5
            numer=0
            for k=1:K
                numer+=rho[isim,t,k]*(cvesim[isim,t,k]-cvesim[isim,s,k])
            end

            denom=@inbounds (vsim[isim,t]-vsim[isim,s])


             if denom>0
                val1=0<numer/denom ? numer/denom : 0.0
                val1=CUDAnative.pow(val1*1.0,(1/(t-1))*1.0)
                dmin=dmin<val1 ? val1 : dmin

             end
             if denom<0
                 val1=0<numer/denom ? numer/denom : 0.0
                 val1=CUDAnative.pow(val1*1.0,(1/(t-1))*1.0)
                  dmax=dmax>val1 ? val1 : dmax
             end
        end


  end


  dmax=dmax>1 ? 1 : dmax
  Deltac[isim]=dmax > dmin ? (unif1[isim]*(dmax-dmin)+dmin) : dmax
  #Deltac[isim]=(unif1[isim]*(dmax-dmin)+dmin)
  return nothing
end

@cuda threads=167 blocks=numblocks new_deltacu!(dcu,vsimcu,cvesimcu,rhocu,Deltaccu,1,unif1)

    #Deltac[:],Wc[:,:,:],vsimc[:,:],cvesimc[:,:,:]=jumpfun2(d=d,gamma=gamma,Delta=Delta,cvesim=cvesim,vsim=vsim,cve=cve,rho=rho);
@everywhere T=5
Deltac[:],Wc[:,:,:],vsimc[:,:],cvesimc[:,:,:]=jumpwrap2!(dcu,Deltacu,vsimcu,cvesimcu,cvecu,rhocu,Deltaccu,vsimccu,cvesimccu,VCu);
logtrydens=(-sum(sum(rho.*Wc,dims=3).^2,dims=2)+ sum(sum(rho.*W,dims=3).^2,dims=2))[:,1,1]
dum=log.(rand(n)).<logtrydens

    #dum=zeros(n).<ones(n)
@inbounds cvesim[dum,:,:]=cvesimc[dum,:,:]
@inbounds W[dum,:,:]=Wc[dum,:,:]
@inbounds vsim[dum,:]=vsimc[dum,:]
@inbounds Delta[dum,:]=Deltac[dum,:]
Deltacu=cu(Delta)
vsimcu=cu(vsim)
cvesimcu=cu(cvesim)

if r>0
  chainM[:,:,r]=myfun(d=d,gamma=gamma,Delta=Delta,W=W,cve=cve,rho=rho)

end
