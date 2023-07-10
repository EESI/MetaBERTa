#!/bin/bash

# SLURM parameters
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=40:00:00
#SBATCH --mem=40G
#SBATCH --job-name=evaluate
#SBATCH --output=./slurmlog/evaluating.log
#SBATCH --array=0-8

module use /ifs/opt_cuda/modulefiles
module load python/gcc/3.10
module load cuda11.2/toolkit cuda11.2/blas cuda11.2/fft tensorrt-cuda11.2 cutensor-cuda11.2

srun python evaluating.py --rowid $1 --table_name $2 --db_file_path $3 --slurmid $SLURM_ARRAY_TASK_ID


# SQL update statement
update_query="UPDATE '$2' SET current_status = 'done' WHERE rowid=$1;"
sqlite3 "$3" "$update_query"

update_query="UPDATE '$2' SET current_state = 'evaluator' WHERE rowid=$1;"
sqlite3 "$3" "$update_query"
