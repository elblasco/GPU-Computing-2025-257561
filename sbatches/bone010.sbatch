#!/usr/bin/env bash

#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --nodelist=edu01
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=bone010
#SBATCH --output=stdout-bone010.txt
#SBATCH --error=stderr-bone010.txt

module load CUDA

srun ./build/output.exec ./dataset/bone010.mtx
