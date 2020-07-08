function dgp12(ri,dlow,n,DP)
(n0,T,K)=size(DP)
Random.seed!(123*ri)
rho=DP[rand(1:n0,n),:,:]
deltasim=rand(n).*(1-dlow).+dlow; lambda=randexp(n)/1
sl=1/15; su=100;
sigma=rand(n,K)*(su-sl) .+ sl
##Multiplicative Error
adum=0.97; bdum=1.03;
epsilon=rand(n,T,K)*(bdum-adum) .+ adum
cve=zeros(n,T,K)
for i=1:n, t=1:T, k=1:K
      cve[i,t,k]= ((lambda[i]/deltasim[i]^(t-1))*rho[i,t,k]).^(-1/sigma[i,k])*epsilon[i,t,k]
end
return rho, cve/1e5
end
