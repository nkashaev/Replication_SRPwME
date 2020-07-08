dlow=theta0
deltasim=rand(n).*(1-dlow).+dlow
lambda=ones(n,1)
lambda=randexp(n)/1
#lambda=ones(n)
#psit=hcat(hcat(ones(n).*1,ones(n).*100),hcat(ones(n).*200,ones(n).*300))
psit=hcat(hcat(ones(n).*1,ones(n).*2),hcat(ones(n).*1,ones(n).*2))
su=100
sl=1/15
sigma=rand(n,K)*(su-sl) .+ sl

##Multiplicative Error
adum=0.97
bdum=1.03
 # adum=1.0
 # bdum=1.0
epsilon=adum .+ rand(n,T,K)*(bdum-adum)
for i=1:n
   for t=1:T
     for k=1:K
       cve[i,t,k]= beta<1 ? ((lambda[i]/deltasim[i]^(t-1))*rho[i,t,k]*psit[i,t]).^(-1/sigma[i,k])*epsilon[i,t,k] : ((lambda[i]/deltasim[i]^(t-1))*rho[i,t,k]).^(-1/sigma[i,k])*epsilon[i,t,k]
     end
   end
 end

cve=cve/1e7
