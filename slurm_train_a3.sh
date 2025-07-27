#!/bin/bash
#SBATCH --job-name=pretrain_study13
#SBATCH --output=logs/train_a3_%j.out
#SBATCH --error=logs/train_a3_%j.err
#SBATCH --time=48:00:00
#SBATCH --account=fuge-prj-jrl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Load modules
module load cuda/12.3.0/gcc/11.3.0/x86_64
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-x86_64

# Activate virtual environment
source /home/gapaza/scratch/repos/diffusion-top-rl/nvenv/bin/activate

# Change directory
cd /home/gapaza/scratch/repos/generative-chess

# Run the training script
python3 -m train_a3