#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=23:00:00
#SBATCH --job-name THF57_580_pilot
#SBATCH --output=/gpfs/fs0/scratch/v/vaguiar/vaguiar/results/TH_580output.txt
module load r/3.4.3-anaconda5.1.0
module load julia/1.1.0
julia /gpfs/fs1/home/v/vaguiar/vaguiar/ReplicationAK/SecondApp/tremblinghand/2App_th_580.jl -p 20