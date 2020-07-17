###########################################################
### Main functions
###########################################################

## Centering Moments Function
##myfun
#d.- discount factor, here it is set to 1.
#gamma.- passing zero matrix of the (dg x 1) size
#eta.- passes the simulated data, here it is equal to the simulated c^* (true consumption) satisfying GARP given quantities and budget constraints
#U.- When active it passes utility numbers from the Afriat inequalities, here it is not active.
#W.- passes zero array to be filled with the simulated error, here it is filled by the moments per each individual observation
#gvec.- passes zero vector to be filled the average of moments
#dummf.- auxiliary zero vector for the algorith; here not active
#cve.- consumption array
#rho.- effective price array
@everywhere function myfun(;d=d::Float64,gamma=gamma::Float64,eta=eta::Float64,U=U::Float64,W=W::Float64,gvec=gvec::Array{Float64,2},dummf=dummf::Float64,cve=cve::Float64,rho=rho::Float64)
    W[:]=cve-reshape(eta,n,T,K)
    gvec[:,:]=reshape(W,n,T*K)/1000
    gvec
end
