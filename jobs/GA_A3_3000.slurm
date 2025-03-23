#!/bin/bash
#SBATCH --job-name="GA_A3_3000"
#SBATCH --partition=medium
#SBATCH --nodes=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/GA_A3_3000-%j.o"
#SBATCH --error="jobs_oe/GA_A3_3000-%j.e"
echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python test_nsga.py --generation 3000 --dataset chosen_0 --start fixed --pop_size 100 --model_name A3
