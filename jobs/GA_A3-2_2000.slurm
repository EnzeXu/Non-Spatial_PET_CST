#!/bin/bash
#SBATCH --job-name="GA_A3-2_2000"
#SBATCH --partition=medium
#SBATCH --nodes=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/GA_A3-2_2000-%j.o"
#SBATCH --error="jobs_oe/GA_A3-2_2000-%j.e"
echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python test_nsga.py --model_name A3-2 --generation 2000 --dataset chosen_0 --start fixed --pop_size 100 --option option2
