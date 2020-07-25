function ED_det_test(rho,cve,stepdum)
    n,T,K=size(rho)
    ind=1
    q=cve[ind,:,:]'
    sol = linprog([-1,0],[2 1],'<',1.5, ClpSolver())
    solsuccess=sol.status
    deltat=collect(.1:stepdum:1)
    soldet=zeros(size(deltat,1),n)
    A0=zeros(size(q,2),size(q,2),size(q,2))
    b0=zeros(size(q,2),size(q,2),1)

    for ind=1:n
      for dd=1:size(deltat,1)
        (Ar,br)=lccons(delta=deltat[dd],p=rho[ind,:,:]',q=cve[ind,:,:]',A=A0,b=b0)
        Ar=reshape(Ar,size(q,2)*size(q,2),size(q,2))
        br=reshape(br,size(q,2)*size(q,2),1)
        ind0=zeros(size(Ar,1))

        for i=1:size(Ar,1)
          if Ar[i,:]==zeros(size(Ar,2))
            ind0[i]=false
          else
            ind0[i]=true
          end
        end

        ind0=ind0.==ones(size(Ar,1))
        Ar=Ar[ind0,:]
        br=br[ind0,:]
        c=zeros(T)
        lb=ones(T)
        ub=Inf*ones(T)
        sol = linprog(c,Ar,'<',br[:,1],lb,ub,ClpSolver())
        soldet[dd,ind]=sol.status==solsuccess
      end
    end

    ## Rejection Rate
    rate=1-sum((sum(soldet,dims=1)[:].>=ones(n))*1)/n
return rate
end

function lccons(;delta=delta,p=p,q=q,A=A,b=b)
  for i=1:size(q,2)
  tmp1=-Matrix{Float64}(I, size(q,2), size(q,2))
  tmp1[:,i]=ones(size(q,2),1)
  tmp1[i,:]=zeros(1,size(q,2))
  tmp2=zeros(size(q,2),1)
    for j=1:size(q,2)
      tmp2[j,1]=(delta^(-j+1)*(p[:,j]'*(q[:,i]-q[:,j])))[1]
    end
  A[i,:,:]=tmp1
  b[i,:,:]=tmp2
  end
  (A,b)
end
