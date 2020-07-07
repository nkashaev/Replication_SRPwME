dlow=.8
deltasim=rand(n).*(1-dlow).+dlow
lambda=randexp(n)/1
psit=hcat(hcat(ones(n).*2,ones(n).*1),hcat(ones(n).*2,ones(n).*1))
su=100
sl=1/15
sigma=rand(n,K)*(su-sl) .+ sl
##Multiplicative Error
adum=0.97
bdum=1.03
epsilon=adum .+ rand(n,T,K)*(bdum-adum)
@simd for i=1:n
   for t=1:T
     for k=1:K
       cve[i,t,k]= ((lambda[i]/deltasim[i]^(t-1))*rho[i,t,k]).^(-1/sigma[i,k])*epsilon[i,t,k]
     end
   end
 end

cve=cve/1e5
