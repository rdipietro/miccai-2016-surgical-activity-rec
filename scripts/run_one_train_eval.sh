#!/bin/bash -l

#SBATCH
#SBATCH --job-name=gpu
#SBATCH --time=0-16:0:0
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G

echo
echo "Running $@ on $SLURMD_NODENAME ..."
echo

# Arguments are used to name the output notebook.
jupyter nbconvert ../train_and_eval.ipynb --to notebook --output "trained $@" --execute --ExecutePreprocessor.timeout=-1

