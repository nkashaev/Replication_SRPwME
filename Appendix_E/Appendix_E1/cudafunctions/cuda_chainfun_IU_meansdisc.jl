using CuArrays
using CuArrays.CURAND
using CUDAnative
using CUDAdrv

function new_lambdacu!(d,Lambda,vsim,cvesim,rho,Lambdac,isim,unif1)
    Lambdac[isim,1]=1.0
    for t=2:T
        thetamin=0.9000001
        thetamax=1.100000*1.0

        for s=1:T
            denom=0
            for k=1:K
                denom+=rho[isim,t,k]*(cvesim[isim,t,k]-cvesim[isim,s,k])
            end

            numer=@inbounds (vsim[isim,t]-vsim[isim,s])

            val1=numer/denom

             if denom<0
                thetamin=thetamin<val1 ? val1 : thetamin

             end
             if denom>0
                  thetamax=thetamax>val1 ? val1 : thetamax
             end
        end
        Lambdac[isim,t]=unif1[isim,t]*(thetamax-thetamin)+thetamin
  end


 
  return nothing
end


##New vsim and cvesim generator
## desc: takes as arguments consistent values of delta and then genertes new

function new_VCcu!(VC,P,dVC,Lambda,vsimc,cvesimc,isim,unif2)
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
               output_denom1=-CUDAnative.pow(Lambda[isim,j]*1.0,(-1)*1.0)*(dVC[isim,j,1]-dVC[isim,i,1])
               output_num1=CUDAnative.pow(Lambda[isim,j]*1.0,(-1)*1.0)*(VC[isim,j,1]-VC[isim,i,1])
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



function jumpfuncu!(d,Lambda,vsim,cvesim,rho,Lambdac,vsimc,cvesimc,unif1,unif2,VC,dVC)


  index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

  stride = blockDim().x * gridDim().x

  for isim = index:stride:n
          new_lambdacu!(d,Lambda,vsim,cvesim,rho,Lambdac,isim,unif1)
          ##
          for t=1:T
              VC[isim,t,1]=vsim[isim,t]
           for k=1:K
               VC[isim,t,k+1]=cvesim[isim,t,k]
           end
          end
          ##
          new_VCcu!(VC,rho,dVC,Lambdac,vsimc,cvesimc,isim,unif2)
 end

  return nothing
end
################################################################################
#######################################################################################



function jumpwrap2!(d,Lambda,vsim,cvesim,cve,rho,Lambdac,vsimc,cvesimc,VC)
    unif1=CuArrays.rand(n,T)
    v=CuArrays.rand(n,T,(K+1))
    dVC=v./norm(v)
    unif2=CuArrays.rand(n)
    numblocks = ceil(Int, n/167)
    @cuda threads=167 blocks=numblocks jumpfuncu!(d,Lambda,vsim,cvesim,rho,Lambdac,vsimc,cvesimc,unif1,unif2,VC,dVC)
    return Array(Lambdac), Array(cve-cvesimc), Array(vsimc), Array(cvesimc)

end;


####################################################################################
###################################################################################
function gchaincu!(d,gamma,cve,rho,chainM=chainM)
    dcu=cu(d)
    Lambdac=zeros(n,T)
    Deltac=zeros(n)
    Wc=ones(n,T,K)
    cvesimc=zeros(n,T,K)
    vsimc=zeros(n,T)
    ##cuda
    Lambdacu=cu(Lambda)
    Deltacu=cu(Delta)
    vsimcu=cu(vsim)
    cvesimcu=cu(cvesim)
    ##candidates
    Lambdaccu=cu(Lambdac)
    Deltaccu=cu(Deltac)
    vsimccu=cu(vsimc)
    cvesimccu=cu(cvesimc)
    ## auxiliaries
    cvecu=cu(cve)
    rhocu=cu(rho)
    VCu=cu(zeros(n,T,K+1))




    r=-repn[1]+1
    while r<=repn[2]
      Lambdac[:,:],Wc[:,:,:],vsimc[:,:],cvesimc[:,:,:]=jumpwrap2!(dcu,Lambdacu,vsimcu,cvesimcu,cvecu,rhocu,Lambdaccu,vsimccu,cvesimccu,VCu);
      logtrydens=(-sum(sum(rho.*Wc,dims=3).^2,dims=2)+ sum(sum(rho.*W,dims=3).^2,dims=2))[:,1,1]
      logtrydens2=(Delta[:].^(1).*Lambdac[:,2] .-1).^2+(Delta[:].^(2).*Lambdac[:,3] .-1).^2+(Delta[:].^(3).*Lambdac[:,4] .-1).^2
      logtrydens3=(Delta[:].^(1).*Lambda[:,2] .-1).^2+(Delta[:].^(2).*Lambda[:,3] .-1).^2+(Delta[:].^(3).*Lambda[:,4] .-1).^2
      dum=log.(rand(n)).<logtrydens-logtrydens2+logtrydens3

      @inbounds cvesim[dum,:,:]=cvesimc[dum,:,:]
      @inbounds W[dum,:,:]=Wc[dum,:,:]
      @inbounds vsim[dum,:]=vsimc[dum,:]
      @inbounds Lambda[dum,:]=Lambdac[dum,:]
      Lambdacu=cu(Lambda)
      vsimcu=cu(vsim)
      cvesimcu=cu(cvesim)
      if r>0
        chainM[:,:,r]=myfun(d=d,gamma=gamma,Lambda=Lambda,Delta=Delta,W=W,cve=cve,rho=rho)

      end
      r=r+1
    end

end
