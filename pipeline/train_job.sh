#!/bin/bash
# SLURM parameters
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --time=120:00:00
#SBATCH --mem=50G
#SBATCH --job-name=training
#SBATCH --output=./slurmlog/training.log

module use /ifs/opt_cuda/modulefiles
module load python/gcc/3.10
module load cuda11.2/toolkit cuda11.2/blas cuda11.2/fft tensorrt-cuda11.2 cutensor-cuda11.2



python training.py --rowid $1 --table_name $2 --db_file_path $3




