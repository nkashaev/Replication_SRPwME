#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --time=23:00:00
#SBATCH --job-name THv3
#SBATCH --output=/gpfs/fs0/scratch/v/vaguiar/vaguiar/results/output.txt
module load r/3.4.3-anaconda5.1.0
module load julia/1.1.0
julia /gpfs/fs1/home/v/vaguiar/vaguiar/ReplicationAK/SecondApp/AK_experimental_trembling_hand_original_niagara.jl -p 20