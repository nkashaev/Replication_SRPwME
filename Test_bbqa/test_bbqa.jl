using LinearAlgebra
using Random
using MathProgBase
using Clp
using DataFrames
using CSV
using NLopt
using BlackBoxOptim

 global count1=0

guessgamma=[-1.0,1.0, 1.0]
dg=length(guessgamma)
opt=NLopt.Opt(:LN_BOBYQA,dg)
toluser=1e-2
NLopt.lower_bounds!(opt,ones(dg).*-Inf)
NLopt.upper_bounds!(opt,ones(dg).*Inf)
NLopt.xtol_rel!(opt,toluser)
NLopt.min_objective!(opt,objMCcu)
(minf,minx,ret) = NLopt.optimize!(opt, guessgamma)
println(minf)
println(minx)
println(ret)
println(count)

function objMCcu(gamma0::Vector, grad::Vector)
  if length(grad) > 0
  end
  global count1
    count1::Int += 1
    println("f_$count1($gamma0)")

  return 100.0 * (gamma0[2] - gamma0[1]^2)^2 + (1.0 - gamma0[1])^2+(gamma0[3] - gamma0[1]^2)^2
end
