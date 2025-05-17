#!/usr/bin/env bash

#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --nodelist=edu01
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=dielFilterV3real
#SBATCH --output=stdout-dielFilterV3real.txt
#SBATCH --error=stderr-dielFilterV3real.txt

module load CUDA

srun ./build/gpu-output.exec ./dataset/dielFilterV3real.mtx
