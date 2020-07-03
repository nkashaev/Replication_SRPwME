###########################################################
### Main functions
###########################################################
## Moments Function
##Fast myfun
@everywhere function myfun(;d=d::Float64,gamma=gamma::Float64,eta=eta::Float64,U=U::Float64,W=W::Float64,gvec=gvec::Array{Float64,2},dummf=dummf::Float64,cve=cve::Float64,rho=rho::Float64)
    W[:]=cve-reshape(eta,n,T,K)
    gvec[:,:]=reshape(W,n,T*K)/1000
    gvec
end
