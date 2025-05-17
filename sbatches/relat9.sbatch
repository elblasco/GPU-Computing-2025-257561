#!/usr/bin/env bash

#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --nodelist=edu01
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=relat9
#SBATCH --output=stdout-relat9.txt
#SBATCH --error=stderr-relat9.txt

module load CUDA

srun ./build/gpu-output.exec ./dataset/relat9.mtx
