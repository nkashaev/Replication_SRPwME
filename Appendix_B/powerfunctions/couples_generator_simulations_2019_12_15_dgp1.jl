dlow=theta0
deltasim=rand(n).*(1-dlow).+dlow
dlowb=.99
dhighb=.99
deltasimb=rand(n).*(dhighb-dlowb).+dlowb
lambda=randexp(n)/1
lambdab=randexp(n)/1
# lambda=ones(n)
# lambdab=ones(n)
#### Pareto Weights
mulo=1/3
muhi=2/3
mu=rand(n,T,K).*(muhi-mulo).+mulo
#lambda=ones(n)
#mut=hcat(hcat(rand(n),rand(n).^2),hcat(rand(n).^3,rand(n).^4))
#mut=hcat(hcat(ones(n).*2/3,ones(n).*2/3),hcat(ones(n).*2/3,ones(n).*2/3))
psit=hcat(hcat(ones(n).*1,ones(n).*100),hcat(ones(n).*200,ones(n).*300))
su=100
sl=1/15
sigma=rand(n,K)*(su-sl) .+ sl
sigmab=rand(n,K)*(su-sl) .+ sl
#sigmab=ones(n,K).* .5
##Multiplicative Error

##
adum=0.97
bdum=1.03
# adum=1
# bdum=1
rhoa=randexp(n,T,K)/2
rhob=randexp(n,T,K)/2
epsilon=adum .+ rand(n,T,K)*(bdum-adum)
@simd for i=1:n
   for t=1:T
     for k=1:K


      # cvea=((lambda[i]/deltasim[i]^(t-1))*rhoa[i,t,k]).^(-1/sigma[i,k])
      # cveb=((lambdab[i]/deltasimb[i]^(t-1))*rhob[i,t,k]).^(-1/sigmab[i,k])

       #cvea=((lambda[i]/deltasim[i]^(t-1))*mu[i,t,k]*rho[i,t,k]).^(-1/sigma[i,k])
       #cveb=((lambdab[i]/deltasimb[i]^(t-1))*(1-mu[i,t,k])*rho[i,t,k]).^(-1/sigmab[i,k])
       # rhoadum=minimum([rhoa[i,t,k] rhoold[i,t,k]*.50])
       # rhobdum=rhoold[i,t,k]-rhoadum
       rhoadum=mu[i,t,k]*rhoold[i,t,k]
       rhobdum=rhoold[i,t,k]-rhoadum
       #cveb=((lambdab[i]*(2/3)/deltasimb[i]^(t-1))*rhobdum).^(-1/sigmab[i,k])
       # cvea=((lambda[i]*(1/3)/deltasim[i]^(t-1))*rhoadum).^(-1/sigma[i,k])
       # cveb=((lambdab[i]*(2/3)/deltasimb[i]^(t-1))*rhobdum).^(-1/sigmab[i,k])
       cvea=((lambda[i]/deltasim[i]^(t-1))*rhoadum).^(-1/sigma[i,k])
       cveb=((lambdab[i]/deltasimb[i]^(t-1))*rhobdum).^(-1/sigmab[i,k])

       cve[i,t,k]=  (cvea+cveb)*epsilon[i,t,k]
       #rho[i,t,k]=rhoa[i,t,k]+rhob[i,t,k]
       rho[i,t,k]=rhoadum+rhobdum
     end
   end
 end

cve=cve/2e6
rho
