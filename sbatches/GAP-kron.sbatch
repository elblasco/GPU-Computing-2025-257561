#!/usr/bin/env bash

#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=gpu-computing-assignment-GAP-kron
#SBATCH --output=stdout-GAP-kron.txt
#SBATCH --error=stderr-GAP-kron.txt
#SBATCH --mail-user=alessio.blascovich@studenti.unitn.it
#SBATCH --mail-type=TIME_LIMIT,END,FAIL

module load CUDA

srun ./build/gpu-output.exec ./dataset/GAP-kron.mtx
