# using LinearAlgebra
# using Random
# using MathProgBase
# using Clp
# using DataFrames
# using CSV
using NLopt
using OptimPack
#using BlackBoxOptim
function objMCcu(gamma0::Vector, grad::Vector)
  if length(grad) > 0
  end
  #global count1
    #count1::Int += 1

  return 100.0 * (gamma0[2] - gamma0[1]^2)^2 + (1.0 - gamma0[1])^2+(gamma0[3] - gamma0[1]^2)^2
end

function obj(gamma0::Vector)

  return 100.0 * (gamma0[2] - gamma0[1]^2)^2 + (1.0 - gamma0[1])^2+(gamma0[3] - gamma0[1]^2)^2
end

dg=3
bdl = -1.0
bdu =  1.0
xl=ones(dg).*bdl
xu=ones(dg).*bdu
rhobeg = 0.1
rhoend = 1e-6
n0=length(x)
npt = 2*n0 + 1
x=zeros(dg)

res=OptimPack.Powell.bobyqa!(obj, x, xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=2, maxeval=5000000)
res[[3]]
x

res2=OptimPack.Powell.bobyqa!(obj,x , xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=2, maxeval=500000)


global count1=0
toluser=1e-10
guessgamma=[-1.0,1.0, 1.0]
dg=length(guessgamma)
opt=NLopt.Opt(:LN_BOBYQA,dg)
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
println(minf)

global count1=0

guessgamma=[-1.0,1.0, 1.0]
dg=length(guessgamma)
opt=NLopt.Opt(:LN_BOBYQA,dg)
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
println(minf)

global count1=0

guessgamma=[-1.0,1.0, 1.0]
dg=length(guessgamma)
opt=NLopt.Opt(:LN_BOBYQA,dg)
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
println(minf)
guessgamma=minx
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
println(minf)
guessgamma=minx
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
println(minf)
guessgamma=minx
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
println(minf)
println(NLOPT_VERSION)

# guessgamma=[-1.0,1.0, 1.0]
# (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
# println(minf)
# guessgamma=minx
# (minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
# println(minf)
#
# println(count1)
