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


function dgp34(ri,mu_par,n,DP)
#mu_par=[0.5, 0.5] or mu=[1/3, 2/3]
(n0,T,K)=size(DP)
Random.seed!(123*ri)
rho=DP[rand(1:n0,n),:,:]
dlowa=.1; dlowb=.99;
deltasima=rand(n).*(1-dlowa) .+dlowa
deltasimb=rand(n).*(1-dlowb) .+dlowb
lambda=randexp(n); lambdab=randexp(n)
#### Pareto Weights
mu=rand(n,T,K).*(mu_par[2]-mu_par[1]) .+ mu_par[1]
sl=1/15; su=100;
sigma=rand(n,K)*(su-sl) .+ sl
sigmab=rand(n,K)*(su-sl) .+ sl

##
adum=0.97; bdum=1.03;
epsilon=rand(n,T,K)*(bdum-adum) .+ adum
cve=zeros(n,T,K)
@simd for i=1:n, t=1:T, k=1:K
       rhoadum=mu[i,t,k]*rho[i,t,k]
       rhobdum=rho[i,t,k]-rhoadum
       cvea=((lambda[i]/deltasima[i]^(t-1))*rhoadum).^(-1/sigma[i,k])
       cveb=((lambdab[i]/deltasimb[i]^(t-1))*rhobdum).^(-1/sigmab[i,k])
       cve[i,t,k]=  (cvea+cveb)*epsilon[i,t,k]
     end

return rho, cve/1e7
end
