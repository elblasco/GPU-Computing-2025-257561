#!/usr/bin/env bash

#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=gpu-computing-assignment-mycielskian17
#SBATCH --output=stdout-mycielskian17.txt
#SBATCH --error=stderr-mycielskian17.txt
#SBATCH --mail-user=alessio.blascovich@studenti.unitn.it
#SBATCH --mail-type=TIME_LIMIT,END,FAIL

module load CUDA

srun ./build/gpu-output.exec ./dataset/mycielskian17.mtx
