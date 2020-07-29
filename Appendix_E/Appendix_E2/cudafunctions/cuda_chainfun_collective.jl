using CuArrays
using CuArrays.CURAND
using CUDAnative
using CUDAdrv

##New delta function:
## desc: passes the discount factors from the cpu to the gpu

function new_delta_collecive_cu!(d,DeltaA,DeltaB,vsimA,vsimB,cvesim,rho,DeltacA,DeltacB,isim,unif1)
  DeltacA[isim]=DeltaA[isim]*1.0
  DeltacB[isim]=DeltaB[isim]*1.0
  return nothing
end

##New vsim and cvesim generator
## desc: takes as arguments consistent values of delta and then genertes new values of consumption satisfying collective inequalities
##Collective VCU
function new_VC_collective_cu!(VC,P,dVC,DeltaA,DeltaB,vsimcA,vsimcB,cvesimc,isim,unif2)
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
               output_denom1=-CUDAnative.pow(DeltaA[isim]*1.0,(j-1)*1.0)*(dVC[isim,j,1]-dVC[isim,i,1])-CUDAnative.pow(DeltaB[isim]*1.0,(j-1)*1.0)*(dVC[isim,j,2]-dVC[isim,i,2])
               output_num1=CUDAnative.pow(DeltaA[isim]*1.0,(j-1)*1.0)*(VC[isim,j,1]-VC[isim,i,1])+CUDAnative.pow(DeltaB[isim]*1.0,(j-1)*1.0)*(VC[isim,j,2]-VC[isim,i,2])
               for k in 1:K
                   output_denom1+=P[isim,j,k]*(dVC[isim,j,k+2]-dVC[isim,i,k+2])
                   output_num1+=-(P[isim,j,k]*(VC[isim,j,k+2]-VC[isim,i,k+2]))
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
       vsimcA[isim,j]=VC[isim,j,1]+newdir*dVC[isim,j,1]
       vsimcB[isim,j]=VC[isim,j,2]+newdir*dVC[isim,j,2]
     for i=3:(K+2)
         cvesimc[isim,j,i-2]=VC[isim,j,i]+newdir*dVC[isim,j,i]
     end
   end
  return nothing
end;



################################################################################
#######################################################################################
function jumpfun_collective_cu!(d,DeltaA,DeltaB,vsimA,vsimB,cvesim,rho,DeltacA,DeltacB,vsimcA,vsimcB,cvesimc,unif1,unif2,VC,dVC)

  index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

  stride = blockDim().x * gridDim().x

  for isim = index:stride:n
          new_delta_collecive_cu!(d,DeltaA,DeltaB,vsimA,vsimB,cvesim,rho,DeltacA,DeltacB,isim,unif1)
          ##
          for t=1:T
              VC[isim,t,1]=vsimA[isim,t]
              VC[isim,t,2]=vsimB[isim,t]
           for k=1:K
               VC[isim,t,k+2]=cvesim[isim,t,k]
           end
          end
          ##
          new_VC_collective_cu!(VC,rho,dVC,DeltacA,DeltacB,vsimcA,vsimcB,cvesimc,isim,unif2)
 end

  return nothing
end

#######################################################################################3




#################################################################################
#################################################################################
function jumpwrap2_collective!(d,DeltaA,DeltaB,vsimA,vsimB,cvesim,cve,rho,DeltacA,DeltacB,vsimcA,vsimcB,cvesimc,VC)
    unif1=CuArrays.rand(n)
    v=CuArrays.rand(n,T,(K+2))
    dVC=v./norm(v)
    unif2=CuArrays.rand(n)
    numblocks = ceil(Int, n/167)
    @cuda threads=167 blocks=numblocks jumpfun_collective_cu!(d,DeltaA,DeltaB,vsimA,vsimB,cvesim,rho,DeltacA,DeltacB,vsimcA,vsimcB,cvesimc,unif1,unif2,VC,dVC)
    return Array(DeltacA), Array(DeltacB), Array(cve-cvesimc), Array(vsimcA), Array(vsimcB), Array(cvesimc)

end;

###



##################################################################################
#################################################################################
function gchain_collective_cu!(d,gamma,cve,rho,chainM=chainM)
    dcu=cu(d)
    DeltacA=zeros(n)
    DeltacB=zeros(n)
    Wc=ones(n,T,K)
    cvesimc=zeros(n,T,K)
    vsimcA=zeros(n,T)
    vsimcB=zeros(n,T)
    ##cuda
    DeltacuA=cu(darandsim)
    DeltacuB=cu(dbrandsim)

    vsimcuA=cu(vsimA)
    vsimcuB=cu(vsimB)
    cvesimcu=cu(cvesim)
    ##candidates
    DeltaccuA=cu(DeltacA)
    DeltaccuB=cu(DeltacB)
    vsimccuA=cu(vsimcA)
    vsimccuB=cu(vsimcB)
    cvesimccu=cu(cvesimc)
    ## auxiliaries
    cvecu=cu(cve)
    rhocu=cu(rho)
    VCu=cu(zeros(n,T,K+2))




    r=-repn[1]+1
    while r<=repn[2]
      DeltacA[:],DeltacB[:],Wc[:,:,:],vsimcA[:,:],vsimcB[:,:],cvesimc[:,:,:]=jumpwrap2_collective!(dcu,DeltacuA,DeltacuB,vsimcuA,vsimcuB,cvesimcu,cvecu,rhocu,DeltaccuA,DeltaccuB,vsimccuA,vsimccuB,cvesimccu,VCu);
      logtrydens=(-sum(sum(rho.*Wc,dims=3).^2,dims=2)+ sum(sum(rho.*W,dims=3).^2,dims=2))[:,1,1]
      dum=log.(rand(n)).<logtrydens

      @inbounds cvesim[dum,:,:]=cvesimc[dum,:,:]
      @inbounds W[dum,:,:]=Wc[dum,:,:]
      @inbounds vsimA[dum,:]=vsimcA[dum,:]
      @inbounds vsimB[dum,:]=vsimcB[dum,:]
      @inbounds DeltaA[dum,:]=DeltacA[dum,:]
      @inbounds DeltaB[dum,:]=DeltacB[dum,:]
      DeltacuA=cu(DeltaA)
      DeltacuB=cu(DeltaB)
      vsimcuA=cu(vsimA)
      vsimcuB=cu(vsimB)
      cvesimcu=cu(cvesim)
      if r>0
        chainM[:,:,r]=myfun(d=d,gamma=gamma,DeltaA=DeltaA,DeltaB=DeltaB,W=W,cve=cve,rho=rho)

      end
      r=r+1
    end

end
