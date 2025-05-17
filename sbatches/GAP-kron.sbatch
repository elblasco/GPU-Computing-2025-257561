#!/usr/bin/env bash

#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --nodelist=edu01
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=GAP-kron
#SBATCH --output=stdout-GAP-kron.txt
#SBATCH --error=stderr-GAP-kron.txt

module load CUDA

srun ./build/gpu-output.exec ./dataset/GAP-kron.mtx
