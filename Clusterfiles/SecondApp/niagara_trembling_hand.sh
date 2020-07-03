#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=23:00:00
#SBATCH --job-name THF57_900_pilot
#SBATCH --output=/gpfs/fs0/scratch/v/vaguiar/vaguiar/results/output.txt
module load r/3.4.3-anaconda5.1.0
module load julia/1.1.0
julia /gpfs/fs1/home/v/vaguiar/vaguiar/ReplicationAK/SecondApp/tremblinghand/AK_Footnote_57_experimental_trembling_hand_reps_580.jl -p 20