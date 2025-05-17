#!/usr/bin/env bash

#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=gpu-computing-assignment-dielFilterV3real
#SBATCH --output=stdout-dielFilterV3real.txt
#SBATCH --error=stderr-dielFilterV3real.txt
#SBATCH --mail-user=alessio.blascovich@studenti.unitn.it
#SBATCH --mail-type=TIME_LIMIT,END,FAIL

module load CUDA

srun ./build/gpu-output.exec ./dataset/dielFilterV3real.mtx
