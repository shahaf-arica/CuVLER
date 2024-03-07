#!/bin/bash
#SBATCH -p work
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=256G
#SBATCH -o "/home/ssaricha/CutLER/submitit/slurm-%j.out"
#SBATCH -e "/home/ssaricha/CutLER/submitit/slurm-%j.err"


srun tools/single-node_run.sh $@
