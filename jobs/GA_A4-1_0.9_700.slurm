#!/bin/bash
#SBATCH --job-name="GA_A4-1_0.9_700"
#SBATCH --partition=medium
#SBATCH --nodes=2
#SBATCH --time=1-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/GA_A4-1_0.9_700-%j.o"
#SBATCH --error="jobs_oe/GA_A4-1_0.9_700-%j.e"
echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python test_nsga.py --model_name A4-1_0.9 --generation 700 --dataset chosen_0 --start ranged --pop_size 100 --option option1 --tcsf_scaler 0.9
