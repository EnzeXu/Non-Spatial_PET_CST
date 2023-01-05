#!/bin/bash
#SBATCH --job-name="GA_A4_5000"
#SBATCH --partition=medium
#SBATCH --nodes=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/GA_A4_5000-%j.o"
#SBATCH --error="jobs_oe/GA_A4_5000-%j.e"
echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python test_nsga.py --generation 5000 --dataset chosen_0 --start ranged --pop_size 100 --model_name A4
