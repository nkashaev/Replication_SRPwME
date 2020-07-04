######################################################################
## MC
gtry=cu(ones(n,dg))
gamma=cu(ones(dg))
valf=cu(zeros(n))
expvalf=cu(zeros(n))
expvalfg=cu(zeros(n,dg))
function preobjMCcu_nail(gamma,chainMcu,valf,gtry,expvalf,expvalfg)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    stride = blockDim().x * gridDim().x


    for i=index:stride:n
        expvalf[i]=0.0
        for j=1:nfast
            valf[i]=0.0
            for t=1:dg
                gtry[i,t]=chainMcu[i,t,j]
                valf[i]+=gtry[i,t]*gamma[t]
                #valf[2]+=CUDAnative.pow(geta[i,t]*1.0,2.0)-CUDAnative.pow(gtry[i,t]*1.0,2.0)
            end
            expvalf[i]+=CUDAnative.exp(valf[i])/nfast
            for t=1:dg
            gtry[i,t]=chainMcu[i,t,j]
            expvalfg[i,t]+=CUDAnative.exp(valf[i])*gtry[i,t]/nfast
            end
        end
    end
    return nothing
end

function objMCcu_nail(gamma0::Vector, grad::Vector)

  @inbounds gtry[:]=0
  @inbounds valf[:]=0
  expvalf=cu(zeros(n))
  expvalfg=cu(zeros(n,dg))
  gamma=cu(gamma0)
  numblocks = ceil(Int, n/167)
  @cuda threads=167 blocks=numblocks preobjMCcu_nail(gamma,chainMcu,valf,gtry,expvalf,expvalfg)
  expvalfg=Array(expvalfg)*1.0
  expvalf=Array(expvalf)*1.0
  if length(grad) > 0
      grad=sum(expvalfg./repeat(expvalf,1,dg),dims=1)'
  end
  return sum(log.(expvalf))/n
end
