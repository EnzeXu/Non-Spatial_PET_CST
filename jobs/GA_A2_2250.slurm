#!/bin/bash
#SBATCH --job-name="GA_A2_2250"
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --time=0-05:00:00
#SBATCH --mem=16GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/GA_A2_2250-%j.o"
#SBATCH --error="jobs_oe/GA_A2_2250-%j.e"
echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python test_nsga.py --generation 2250 --dataset all --start ranged --pop_size 100
