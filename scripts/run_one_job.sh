#!/bin/bash -l

#SBATCH
#SBATCH --job-name=gpu
#SBATCH --time=0-24:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G

echo
echo "Running $@ on $SLURMD_NODENAME ..."
echo

python ../train_and_summarize.py "$@"
