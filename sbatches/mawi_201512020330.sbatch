#!/usr/bin/env bash

#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --nodelist=edu01
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=mawi_201512020330
#SBATCH --output=stdout-mawi_201512020330.txt
#SBATCH --error=stderr-mawi_201512020330.txt

module load CUDA

srun ./build/gpu-output.exec ./dataset/mawi_201512020330.mtx
