dlowa=.1
deltasima=rand(n).*(1-dlowa).+dlowa
dlowb=.99
dhighb=1.0
deltasimb=rand(n).*(dhighb-dlowb).+dlowb
lambda=randexp(n)/1
lambdab=randexp(n)/1
#### Pareto Weights
mulo=1/2
muhi=1/2
mu=rand(n,T,K).*(muhi-mulo).+mulo
su=100
sl=1/15
sigma=rand(n,K)*(su-sl) .+ sl
sigmab=rand(n,K)*(su-sl) .+ sl

##
adum=0.97
bdum=1.03
epsilon=adum .+ rand(n,T,K)*(bdum-adum)
@simd for i=1:n
   for t=1:T
     for k=1:K


       rhoadum=mu[i,t,k]*rhoold[i,t,k]
       rhobdum=rhoold[i,t,k]-rhoadum
       cvea=((lambda[i]/deltasima[i]^(t-1))*rhoadum).^(-1/sigma[i,k])
       cveb=((lambdab[i]/deltasimb[i]^(t-1))*rhobdum).^(-1/sigmab[i,k])

       cve[i,t,k]=  (cvea+cveb)*epsilon[i,t,k]
       rho[i,t,k]=rhoadum+rhobdum
     end
   end
 end

cve=cve/1e7
