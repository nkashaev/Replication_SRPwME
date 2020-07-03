#!/bin/bash 
#SBATCH --nodes=20
#SBATCH --ntasks=3
#SBATCH --time=23:00:00
#SBATCH --job-name THFAST
#SBATCH --output=/gpfs/fs0/scratch/v/vaguiar/vaguiar/results/outputFAST.txt
module load r/3.4.3-anaconda5.1.0
module load julia/1.1.0
julia /gpfs/fs1/home/v/vaguiar/vaguiar/ReplicationAK/SecondApp/AK_experimental_trembling_hand_FAST_niagara.jl -p 800