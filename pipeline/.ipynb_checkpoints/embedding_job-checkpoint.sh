#!/bin/bash

# SLURM parameters
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=15:00:00
#SBATCH --mem=20G
#SBATCH --job-name=embedding
#SBATCH --output=./slurmlog/embedding.log
#SBATCH --array=0-2

module use /ifs/opt_cuda/modulefiles
module load python/gcc/3.10
module load cuda11.2/toolkit cuda11.2/blas cuda11.2/fft tensorrt-cuda11.2 cutensor-cuda11.2

python embedding.py --rowid $1 --table_name $2 --db_file_path $3 --slurmid $SLURM_ARRAY_TASK_ID


# Database path
query="SELECT evaluate_path,model_name FROM '$2' WHERE rowid=$1;"

# Execute the query and retrieve the paths
read -r path1 path2 <<< $(sqlite3 "$3" "$query")

# Concatenate the paths
final_path="$path1$path2"
final_path=$(echo "$final_path" | tr -d '|')

# Count the number of files ending with ".npy" in the final path
file_count=$(find "$final_path" -maxdepth 1 -type f -name "*.npy" | wc -l)

# Check if there are exactly five files ending with ".npy"
if [ "$file_count" -eq 5 ]; then
    echo "All files are available: $path"

    # SQL update statement
    update_query="UPDATE '$2' SET current_state = 'evaluator' WHERE rowid=$1;"
    sqlite3 "$3" "$update_query"

    # SQL update statement
    update_query="UPDATE '$2' SET current_status = 'fail' WHERE rowid=$1;"
    sqlite3 "$3" "$update_query"
else
    echo "You need to wait for the rest of the runs"
fi
