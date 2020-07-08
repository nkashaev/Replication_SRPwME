dlow=theta0
deltasim=rand(n).*(1-dlow).+dlow
lambda=randexp(n)/1
#lambda=ones(n)
#mut=hcat(hcat(rand(n),rand(n).^2),hcat(rand(n).^3,rand(n).^4))
mut=hcat(hcat(ones(n).*2/3,ones(n).*2/3),hcat(ones(n).*2/3,ones(n).*2/3))
su=.5
sl=.5
sigma=rand(n,K)*(su-sl) .+ sl
##Multiplicative Error
adum=0.97
bdum=1.03
# adum=1
# bdum=1
epsilon=adum .+ rand(n,T,K)*(bdum-adum)
@simd for i=1:n
   for t=1:T
     for k=1:K
       cve[i,t,k]=((lambda[i]/deltasim[i]^(t-1))*rho[i,t,k]*prod((1 .-(1-beta) .* mut[i,1:t]).^(-1)))^(-1/sigma[i,k])*epsilon[i,t,k]
       #cve[i,t,k]=((lambda[i]/deltasim[i]^(t-1))*rho[i,t,k])^(-1/sigma[i,k])*epsilon[i,t,k]
       #cve[i,t,k]=(1/deltasim[i]^(t-1))*rho[i,t,k]*prod((1-(1-beta) .* mut[i,1:t][1]).^(-1))

     end
   end
 end

cve=cve/1e7
