##Version Julia 1.3.1
##Author: Victor Aguiar
##email: vhaguiar@gmail.com
#package mode
using Pkg
Pkg.add("NLopt")
Pkg.add("LinearAlgebra")
Pkg.add("Random")
Pkg.add("MathProgBase")
Pkg.add("Clp")
Pkg.add("DataFrames")
Pkg.add("JuMP")
Pkg.add("Convex")
Pkg.add("ECOS")
Pkg.add("CSV")
Pkg.add("BlackBoxOptim")
Pkg.add("SCS")
Pkg.add("Ipopt")

### CUDA
Pkg.add("CuArrays")
Pkg.add("CUDAnative")
Pkg.add("CUDAdrv")
Pkg.add("SoftGlobalScope")
