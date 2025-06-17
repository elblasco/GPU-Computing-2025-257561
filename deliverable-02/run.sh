#!/usr/bin/env bash

# Always run this script from the root of the repo

source env.sh

echo -e "${GRE}Building $BIN...${NC}"
make clean $BIN

# Setup environment
if [[ -z $SbM_HOME ]]; then
    echo -e "${GRE}Setting up experiments environment...${NC}"
    #if [[ ! -f SbatchMan/sourceFile.sh ]]; then
    echo -e "${GRE}Initializing SLURM configurations...${NC}"
    cd SbatchMan
    ./initEnv.sh
    cd ..
    source SbatchMan/sourceFile.sh
        # 1 node, 1 CPU, 1 GPU, no MPI
	rm -rf "$SbM_SOUT"
	rm -rf "$SbM_METADATA_HOME"
    SbatchMan/newExperiment.sh -p "edu-short" -t 00:04:00 -e "COO_SPMV" -n 1 -c 1 -g 1 -d 1 -w edu02 -b "$BIN" # -a hackaton
    #else
    #    source SbatchMan/sourceFile.sh
    #fi
fi


source SbatchMan/submit.sh
my_hostname=$(${SbM_UTILS}/hostname.sh)

for file in $MTX_PATH/*; do
    echo "----- Testing '$(basename "${file%.*}")' graph -----"
    SbM_submit_function --verbose --expname "$COO_SPMV_$(basename $file)" --binary $BIN -f "$file" -n $ITERATIONS
    echo "JOB ID: ${job_id}"
done
