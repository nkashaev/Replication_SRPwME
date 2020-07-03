#!/bin/bash 
#SBATCH --account=def-vaguiar   # replace this with your own account 
#SBATCH --ntasks=50               # number of MPI processes 
#SBATCH --mem-per-cpu=3048M      # memory; default unit is megabytes 
#SBATCH --time=1-00:15           # time (DD-HH:MM) 
#SBATCH --output=outputgraham.log  
module load julia 
julia /home/vaguiar/ReplicationAK/SecondApp/AK_experimental_mainv4_graham.jl -p 50