#!/usr/bin/env bash

SBATCH_DIR=./sbatches

module purge

module load CUDA

make all

for SBATCH_FILE in "$SBATCH_DIR"/*; do
	echo "Sbatching $SBATCH_FILE"
	sbatch "$SBATCH_FILE"
done

