#!/bin/bash
#SBATCH --job-name=cic_fedOpt
#SBATCH --partition=common
#SBATCH --cpus-per-task=24
#SBATCH --time=14-00:00:0
#SBATCH --mem=48G
#SBATCH --output=./sbatch/slurm-%j.log

project_path=$(pwd)
source $project_path/env/bin/activate
export JOBLIB_TEMP_FOLDER=/tmp

export CUDA_VISIBLE_DEVICES=""

python $project_path/main_fed.py


