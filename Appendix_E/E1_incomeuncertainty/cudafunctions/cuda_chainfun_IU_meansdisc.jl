using CuArrays
using CuArrays.CURAND
using CUDAnative
using CUDAdrv

##New delta function:
## desc: takes as arguments consistent values of vsim and cvesim, given observed rho, to generate  new consistent delta
# function new_deltacu!(d,vsim,cvesim,rho,Deltac,isim,unif1)
#     dmin=d
#     dmax=1
#
#     for t=2:T
#
#         for s=1:T
#             numer=0
#             for k=1:K
#                 numer+=rho[isim,t,k]*(cvesim[isim,t,k]-cvesim[isim,s,k])
#             end
#
#             denom=@inbounds (vsim[isim,t]-vsim[isim,s])
#
#
#              if denom>0
#                 val1=0<numer/denom ? numer/denom : 0.0
#                 val1=CUDAnative.pow(val1*1.0,(1/(t-1))*1.0)
#                 dmin=dmin<val1 ? val1 : dmin
#
#              end
#              if denom<0
#                  val1=0<numer/denom ? numer/denom : 0.0
#                  val1=CUDAnative.pow(val1*1.0,(1/(t-1))*1.0)
#                   dmax=dmax>val1 ? val1 : dmax
#              end
#         end
#
#
#   end
#
#
#   dmax=dmax>1 ? 1 : dmax
#   Deltac[isim]=dmax > dmin ? (unif1[isim]*(dmax-dmin)+dmin) : dmax
#   #Deltac[isim]=(unif1[isim]*(dmax-dmin)+dmin)
#   return nothing
# end
#
# ## New delta
# function new_deltacu!(d,Delta,vsim,cvesim,rho,Deltac,isim,unif1)
#     dmin=d
#     dmax=1
#
#     for t=2:T
#
#         for s=1:T
#             numer=0
#             for k=1:K
#                 numer+=rho[isim,t,k]*(cvesim[isim,t,k]-cvesim[isim,s,k])
#             end
#
#             denom=@inbounds (vsim[isim,t]-vsim[isim,s])
#
#
#              if denom>0
#                 val1=0<numer/denom ? numer/denom : 0.0
#                 val1=CUDAnative.pow(val1*1.0,(1/(t-1))*1.0)
#                 dmin=dmin<val1 ? val1 : dmin
#
#              end
#              if denom<0
#                  val1=0<numer/denom ? numer/denom : 0.0
#                  val1=CUDAnative.pow(val1*1.0,(1/(t-1))*1.0)
#                   dmax=dmax>val1 ? val1 : dmax
#              end
#         end
#
#
#   end
#
#   dmax=dmax>Delta[isim]*1.0 ? dmax : Delta[isim]*1.0
#   dmin=dmin<Delta[isim]*1.0 ? dmin : Delta[isim]*1.0
#   dmax=dmax>1.0 ? 1.0 : dmax
#   Deltac[isim]=dmax > dmin ? (unif1[isim]*(dmax-dmin)+dmin) : dmax
#   #Deltac[isim]=(unif1[isim]*(dmax-dmin)+dmin)
#   return nothing
# end
######################################################################
##!!New
## New Lambda
# lambdamax=zeros(3,1)
# lambdamin=zeros(3,1)
# for j=1:size(VC,2)
#     thetamin=0.
#     thetamax=10^6.
#     for i=1:size(VC,2)
#
#         if j!=i
#             output_denom1 =  dot(P[:,j], VC[:,j] .- VC[:,i])
#             output_num1   =  VC[1,j] .- VC[1,i]
#
#             if output_denom1>0
#                 thetamax=minimum([thetamax, output_num1/output_denom1])
#             else
#                 thetamin=maximum([thetamin, output_num1/output_denom1])
#             end
#         end
#     end
#     lambdamax[j]=thetamax
#     lambdamin[j]=thetamin
# end

function new_lambdacu!(d,Lambda,vsim,cvesim,rho,Lambdac,isim,unif1)
    Lambdac[isim,1]=1.0
    for t=2:T
        thetamin=0.0000001
        thetamax=1000000*1.0

        for s=1:T
            denom=0
            for k=1:K
                denom+=rho[isim,t,k]*(cvesim[isim,t,k]-cvesim[isim,s,k])
            end

            numer=@inbounds (vsim[isim,t]-vsim[isim,s])

            val1=numer/denom

             if denom<0
                #val1=0<numer/denom ? numer/denom : 0.0
                thetamin=thetamin<val1 ? val1 : thetamin

             end
             if denom>0
                 #val1=0<numer/denom ? numer/denom : 1000000.0
                  thetamax=thetamax>val1 ? val1 : thetamax
             end
        end
        #thetamin=thetamin<0.0000001 ? 0.0000001 : thetamin
        #Lambdac[isim,t]=thetamax > thetamin ? (unif1[isim]*(thetamax-thetamin)+thetamin) : thetamax
        #Lambdac[isim,t]=unif1[isim]*(thetamax-thetamin)+thetamin
        Lambdac[isim,t]=unif1[isim,t]*(thetamax-thetamin)+thetamin
  end


  #Deltac[isim]=(unif1[isim]*(dmax-dmin)+dmin)
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
               #output_denom1=-CUDAnative.pow(Delta[isim]*1.0,(j-1)*1.0)*(dVC[isim,j,1]-dVC[isim,i,1])
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
   #newdir=(unif2[isim]*(thetamax-thetamin)+thetamin)
   for j=1:T
       vsimc[isim,j]=VC[isim,j,1]+newdir*dVC[isim,j,1]
     for i=2:(K+1)
         cvesimc[isim,j,i-1]=VC[isim,j,i]+newdir*dVC[isim,j,i]
     end
   end
  return nothing
end;



function jumpfuncu!(d,Lambda,vsim,cvesim,rho,Lambdac,vsimc,cvesimc,unif1,unif2,VC,dVC)

  # index = threadIdx().x    # this example only requires linear indexing, so just use `x`
  #
  # stride = blockDim().x
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


function jumpwrap!(d,Delta,vsim,cvesim,rho,Deltac,vsimc,cvesimc,VC)
    #unif1=curand(n).*(.9-.1).+.1
    unif1=curand(n)
    v=curandn(n,T,(K+1))
    dVC=v./norm(v)
    #unif2=curand(n).*(.9-.1).+.1
    unif2=curand(n)
    @cuda threads=250 jumpfuncu!(d,Delta,vsim,cvesim,rho,Deltac,vsimc,cvesimc,unif1,unif2,VC,dVC)
end


function jumpwrap2!(d,Lambda,vsim,cvesim,cve,rho,Lambdac,vsimc,cvesimc,VC)
    #unif1=curand(n).*(.999-.001).+.001
    unif1=CuArrays.rand(n,T)
    v=CuArrays.rand(n,T,(K+1))
    dVC=v./norm(v)
    #unif2=curand(n).*(.999-.001).+.001
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
      #Deltac[:],Wc[:,:,:],vsimc[:,:],cvesimc[:,:,:]=jumpfun2(d=d,gamma=gamma,Delta=Delta,cvesim=cvesim,vsim=vsim,cve=cve,rho=rho);
      Lambdac[:,:],Wc[:,:,:],vsimc[:,:],cvesimc[:,:,:]=jumpwrap2!(dcu,Lambdacu,vsimcu,cvesimcu,cvecu,rhocu,Lambdaccu,vsimccu,cvesimccu,VCu);
      logtrydens=(-sum(sum(rho.*Wc,dims=3).^2,dims=2)+ sum(sum(rho.*W,dims=3).^2,dims=2))[:,1,1]
      logtrydens2=(Delta[:].^(1).*Lambdac[:,2] .-1).^2+(Delta[:].^(2).*Lambdac[:,3] .-1).^2+(Delta[:].^(3).*Lambdac[:,4] .-1).^2
      logtrydens3=(Delta[:].^(1).*Lambda[:,2] .-1).^2+(Delta[:].^(2).*Lambda[:,3] .-1).^2+(Delta[:].^(3).*Lambda[:,4] .-1).^2
      dum=log.(rand(n)).<logtrydens-logtrydens2+logtrydens3

      #dum=zeros(n).<ones(n)
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





# aiverify3=cu(zeros(n,T,T))
# function verify(aiverify3,vsimccu,Deltaccu,cvesimccu,rhocu)
#     for id=1:n
#         for t=1:T
#             for s=1:T
#                 aiverify3[id,t,s]=vsimccu[id,t]-vsimccu[id,s]
#                 for k=1:K
#                     aiverify3[id,t,s]+=-CUDAnative.pow(Deltaccu[id]*1.0,-(t-1)*1.0)*rhocu[id,t,k]*(cvesimccu[id,t,k]-cvesimccu[id,s,k])
#                 end
#             end
#         end
#     end
#     return nothing
# end
#
# @cuda verify(aiverify3,vsimccu,Deltaccu,cvesimccu,rhocu)
#
# minimum(aiverify3)
