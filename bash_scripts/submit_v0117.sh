#!/bin/bash

for i in $(seq 0.3 0.1 0.6)
do
    for j in 500 600 700 800
    do
        sbatch jobs/GA_A4-1_${i}_${j}.slurm
    done
done
#sbatch jobs/GA_A4-1_0.3_500.slurm
#sbatch jobs/GA_A4-1_0.3_600.slurm
#sbatch jobs/GA_A4-1_0.3_700.slurm
#sbatch jobs/GA_A4-1_0.3_800.slurm
#sbatch jobs/GA_A4-1_0.3_900.slurm
#sbatch jobs/GA_A4-1_0.3_1000.slurm

#sbatch jobs/GA_A3-1_500.slurm
#sbatch jobs/GA_A3-1_1000.slurm
#sbatch jobs/GA_A3-1_1500.slurm
#sbatch jobs/GA_A3-1_2000.slurm

#sbatch jobs/GA_A3-2_500.slurm
#sbatch jobs/GA_A3-2_1000.slurm
#sbatch jobs/GA_A3-2_1500.slurm
#sbatch jobs/GA_A3-2_2000.slurm

#sbatch jobs/GA_A4-1_500.slurm
#sbatch jobs/GA_A4-1_1000.slurm
#sbatch jobs/GA_A4-1_1500.slurm
#sbatch jobs/GA_A4-1_2000.slurm

#sbatch jobs/GA_A4-2_500.slurm
#sbatch jobs/GA_A4-2_1000.slurm
#sbatch jobs/GA_A4-2_1500.slurm
#sbatch jobs/GA_A4-2_2000.slurm