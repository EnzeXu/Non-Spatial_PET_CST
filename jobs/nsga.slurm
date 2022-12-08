#!/bin/bash -l

#SBATCH --job-name="nsga"
#SBATCH --partition=medium
#SBATCH --nodes=1

#SBATCH --time=2-00:00:00
#SBATCH --mem=64GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output="jobs_oe/nsga-%j.o"
#SBATCH --error="jobs_oe/nsga-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python test_nsga.py