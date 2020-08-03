using CuArrays
using CuArrays.CURAND
using CUDAnative
using CUDAdrv

##New delta function:
## desc: takes as arguments consistent values of vsim and cvesim, given observed rho, to generate  new consistent delta
function new_deltacu!(d,Delta,vsim,cvesim,rho,Deltac,isim,unif1)
    dmin=d
    dmax=1

    for t=2:T

        for s=1:T
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

  dmax=dmax>Delta[isim]*1.0 ? dmax : Delta[isim]*1.0
  dmin=dmin<Delta[isim]*1.0 ? dmin : Delta[isim]*1.0
  dmax=dmax>1.0 ? 1.0 : dmax
  Deltac[isim]=dmax > dmin ? (unif1[isim]*(dmax-dmin)+dmin) : dmax

  return nothing
end
##New vsim and cvesim generator
## desc: takes as arguments consistent values of delta and then genertes new

function new_VCcu!(VC,P,dVC,Delta,vsimc,cvesimc,isim,unif2)
  # given the initial matrix VC and prices P
  # the function samples from the polytope uniformly

  thetamin=-10.0^6 #initial upperbound
  thetamax=10.0^6 #initial lowerbound
  #Generating random direction
  # #Box constraints

  #box constraints do not include v numbers, but included without loss of generality
     for i=1:(K+1)
       for j=1:T
           if dVC[isim,j,i]<0
                 thetamax=thetamax < -(VC[isim,j,i]/dVC[isim,j,i]) ? thetamax : -(VC[isim,j,i]/dVC[isim,j,i])
           else
                 thetamin=thetamin > -(VC[isim,j,i]/dVC[isim,j,i]) ? thetamin : -(VC[isim,j,i]/dVC[isim,j,i])

           end
       end
   end
  # #Afriat constraints
   for i=1:T
       for j=1:T
           if j!=i
               output_denom1=-CUDAnative.pow(Delta[isim]*1.0,(j-1)*1.0)*(dVC[isim,j,1]-dVC[isim,i,1])
               output_num1=CUDAnative.pow(Delta[isim]*1.0,(j-1)*1.0)*(VC[isim,j,1]-VC[isim,i,1])
               for k in 1:K
                   output_denom1+=P[isim,j,k]*(dVC[isim,j,k+1]-dVC[isim,i,k+1])
                   output_num1+=-(P[isim,j,k]*(VC[isim,j,k+1]-VC[isim,i,k+1]))
                end

                if output_denom1>0
                    thetamax=thetamax < (output_num1/output_denom1) ? thetamax : (output_num1/output_denom1)
                 else
                    thetamin=thetamin > (output_num1/output_denom1) ? thetamin : (output_num1/output_denom1)
                end
            end
       end
   end
   newdir=thetamax > thetamin ? (unif2[isim]*(thetamax-thetamin)+thetamin) : 0.0

   for j=1:T
       vsimc[isim,j]=VC[isim,j,1]+newdir*dVC[isim,j,1]
     for i=2:(K+1)
         cvesimc[isim,j,i-1]=VC[isim,j,i]+newdir*dVC[isim,j,i]
     end
   end
  return nothing
end;



function jumpfuncu!(d,Delta,vsim,cvesim,rho,Deltac,vsimc,cvesimc,unif1,unif2,VC,dVC)

  # index = threadIdx().x    # this example only requires linear indexing, so just use `x`
  #
  # stride = blockDim().x
  index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

  stride = blockDim().x * gridDim().x

  for isim = index:stride:n
          new_deltacu!(d,Delta,vsim,cvesim,rho,Deltac,isim,unif1)
          ##
          for t=1:T
              VC[isim,t,1]=vsim[isim,t]
           for k=1:K
               VC[isim,t,k+1]=cvesim[isim,t,k]
           end
          end
          ##
          new_VCcu!(VC,rho,dVC,Deltac,vsimc,cvesimc,isim,unif2)
 end

  return nothing
end
################################################################################
#######################################################################################




function jumpwrap2!(d,Delta,vsim,cvesim,cve,rho,Deltac,vsimc,cvesimc,VC)
    unif1=CuArrays.rand(n)
    v=CuArrays.randn(n,T,(K+1))
    dVC=v./norm(v)
    unif2=CuArrays.rand(n)
    numblocks = ceil(Int, n/167)
    @cuda threads=167 blocks=numblocks jumpfuncu!(d,Delta,vsim,cvesim,rho,Deltac,vsimc,cvesimc,unif1,unif2,VC,dVC)
    return Array(Deltac), Array(cve-cvesimc), Array(vsimc), Array(cvesimc)

end;



####################################################################################
###################################################################################
function gchaincu!(d,gamma,cve,rho,chainM,Delta,vsim,cvesim,W)
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




    r=-repn[1]+1
    while r<=repn[2]
      Deltac[:],Wc[:,:,:],vsimc[:,:],cvesimc[:,:,:]=jumpwrap2!(dcu,Deltacu,vsimcu,cvesimcu,cvecu,rhocu,Deltaccu,vsimccu,cvesimccu,VCu);
      logtrydens=(-sum(sum(rho.*Wc,dims=3).^2,dims=2)+ sum(sum(rho.*W,dims=3).^2,dims=2))[:,1,1]
      dum=log.(rand(n)).<logtrydens

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
      r=r+1
    end

end
