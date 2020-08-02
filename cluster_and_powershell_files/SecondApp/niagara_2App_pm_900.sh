#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=23:00:00
#SBATCH --job-name THF55_900_pilot_seedcheck
#SBATCH --output=/gpfs/fs0/scratch/v/vaguiar/vaguiar/results/output.txt
module load r/3.4.3-anaconda5.1.0
module load julia/1.1.0
julia /gpfs/fs1/home/v/vaguiar/vaguiar/ReplicationAK/SecondApp/pricemisperception/2App_pm_900.jl -p 30