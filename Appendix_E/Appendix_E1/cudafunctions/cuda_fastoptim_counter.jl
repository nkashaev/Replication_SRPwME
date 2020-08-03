
geta=cu(ones(n,dg))
gtry=cu(ones(n,dg))
@inbounds geta[:,:]=chainMcu[:,:,1]
dvecM=cu(zeros(n,dg))
logunif=log.(curand(n,nfast))
gamma=cu(ones(dg))
valf=cu(zeros(n))

function preobjMCcu(gamma,chainMcu,valf,geta,gtry,dvecM,logunif)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    stride = blockDim().x * gridDim().x


    for i=index:stride:n
        for j=1:nfast
            valf[i]=0.0
            for t=1:dg
                gtry[i,t]=chainMcu[i,t,j]
                valf[i]+=gtry[i,t]*gamma[t]-geta[i,t]*gamma[t]
            end
            for t=1:dg
                geta[i,t]=logunif[i,j] < valf[i] ? gtry[i,t] : geta[i,t]
                dvecM[i,t]+=geta[i,t]/nfast
            end
        end
    end
    return nothing
end

function objMCcu(gamma0::Vector, grad::Vector)
  if length(grad) > 0
  end
  @inbounds geta[:]=0
  @inbounds gtry[:]=0
  @inbounds geta[:,:]=chainMcu[:,:,1]
  @inbounds logunif[:]=log.(curand(n,nfast))
  dvecM=cu(zeros(n,dg))
  valf[:]=0
  gamma=cu(gamma0)

  numblocks = ceil(Int, n/167)
  @cuda threads=167 blocks=numblocks preobjMCcu(gamma,chainMcu,valf,geta,gtry,dvecM,logunif)
    dvecM=Array(dvecM)*1.0
    dvec=sum(dvecM,dims=1)'/n


    numvar=zeros(dg,dg)
    @simd for i=1:n
        BLAS.syr!('U',1.0/n,dvecM[i,:],numvar)
    end
    var=numvar+numvar'- Diagonal(diag(numvar))-dvec*dvec'
    (Lambda,QM)=eigen(var)
    inddummy=Lambda.>0
    An=QM[:,inddummy]
    dvecdum2=An'*(dvec)
    vardum3=An'*var*An
    Omega2=inv(vardum3)
    Qn2=1/2*dvecdum2'*Omega2*dvecdum2

    return Qn2[1]
end





function objMCcu2c(gamma0)
  @inbounds geta[:]=0
  @inbounds gtry[:]=0
  @inbounds geta[:,:]=chainMcu[:,:,1]
  @inbounds logunif[:]=log.(curand(n,nfast))
  dvecM=cu(zeros(n,dg))
  valf[:]=0
  gamma=cu(gamma0)

  numblocks = ceil(Int, n/167)
  @cuda threads=167 blocks=numblocks preobjMCcu(gamma,chainMcu,valf,geta,gtry,dvecM,logunif)
    dvecMc=Array(dvecM)*1.0
    dvec=sum(dvecMc,dims=1)'/n


    numvar=zeros(dg,dg)
    @simd for i=1:n
        BLAS.syr!('U',1.0/n,dvecMc[i,:],numvar)
    end
    var=numvar+numvar'- Diagonal(diag(numvar))-dvec*dvec'
    (Lambda,QM)=eigen(var)
    inddummy=Lambda.>0
    An=QM[:,inddummy]
    dvecdum2=An'*(dvec)
    vardum3=An'*var*An
    Omega2=inv(vardum3)
    Qn2=1/2*dvecdum2'*Omega2*dvecdum2

    return Qn2[1]
end
