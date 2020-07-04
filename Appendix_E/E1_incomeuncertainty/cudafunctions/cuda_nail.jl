######################################################################
## MC
gtry=cu(ones(n,dg))
gamma=cu(ones(dg))
valf=cu(zeros(n))
expvalf=cu(zeros(n))

function preobjMCcu_nail(gamma,chainMcu,valf,gtry,expvalf)
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
        end
    end
    return nothing
end

function objMCcu_nail(gamma0::Vector, grad::Vector)
  if length(grad) > 0
  end
  @inbounds gtry[:]=0
  expvalf=cu(zeros(n))
  valf[:]=0
  gamma=cu(gamma0)

  numblocks = ceil(Int, n/167)
  @cuda threads=167 blocks=numblocks preobjMCcu(gamma,chainMcu,valf,geta,gtry,dvecM,logunif)
  preobjMCcu_nail(gamma,chainMcu,valf,gtry,expvalf)
    expvalf=Array(expvalf)*1.0
    return sum(log.(expvalf))/n
end
