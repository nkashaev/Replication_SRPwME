count = 0
#Set the number of processors: Change it to the max the computer allows
nprocs=30
using Distributed
addprocs(nprocs)
@everywhere Distributed
@everywhere using Random
@everywhere using NLopt
@everywhere using DataFrames
@everywhere using MathProgBase
using CSV
@everywhere using RCall
@everywhere using LinearAlgebra
## set directory
machine="niagara"
if machine=="niagara"
  rootdir="/gpfs/fs1/home/v/vaguiar/vaguiar/ReplicationAK/SecondApp"
  dir=rootdir*"/data"
  dirresults="/gpfs/fs0/scratch/v/vaguiar/vaguiar/results"
end
#results file
if machine=="nailmachine"
  ##the following line is eliminated in case the installation of Rcall happened correctly and there is only 1 installation of R
  @everywhere ENV["R_HOME"]="C:\\Users\\Nkashaev\\Documents\\R\\R-3.4.4"
  rootdir="C:/Users/Nkashaev/Dropbox/ReplicationAK/SecondApp"
  dir=rootdir*"/data"
  dirresults=rootdir*"/results"
end

## Because the simulations are done using parallel Montecarlo we have nsimps*nprocs draws.
# If needed, one can set burnrate.
burnrate=0
nsimsp=30
## Running the code.
include(rootdir*"/tremblinghand/2App_th_main.jl")

## Saving Output.
totalnsims=nsimsp*nprocs
DFsolv=convert(DataFrame,results)
CSV.write(dirresults*"//2App_th_reps_$totalnsims.csv",DFsolv)
