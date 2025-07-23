#!/usr/bin/env bash

source env.sh

mkdir -p profilation

# Setup environment
if [[ -z $SbM_HOME ]]; then
    echo -e "${PUR}Setting up experiments environment...${NC}"
    #if [[ ! -f SbatchMan/sourceFile.sh ]]; then
    echo -e "${PUR}Initializing SLURM configurations...${NC}"
    cd SbatchMan || exit
    ./initEnv.sh
    cd .. || exit
    source SbatchMan/sourceFile.sh
        # 1 node, 1 CPU, 1 GPU, no MPI
	rm -rf "$SbM_SOUT"
	rm -rf "$SbM_METADATA_HOME"
    SbatchMan/newExperiment.sh -p "edu-short" -t 00:05:00 -e "NSYS_COO_SPMV" -n 1 -c 1 -g 1 -d 1 -w edu02 -b nsys
fi

source SbatchMan/submit.sh
my_hostname=$(${SbM_UTILS}/hostname.sh)

for file in "$MTX_PATH"/*.sbmtx; do
    echo "----- Testing '$(basename "${file%.*}")' graph -----"
    SbM_submit_function --verbose --expname "$NSYS_COO_SPMV_$(basename $file)" --binary nsys profile --trace='cuda,nvtx' --cuda-memory-usage=true --force-overwrite true -o profilation/"profile-$(basename $file)" $BIN -f $file
    echo "JOB ID: ${job_id}"
done

# Setup environment
echo -e "${PUR}Setting up experiments environment...${NC}"
echo -e "${PUR}Initializing SLURM configurations...${NC}"
cd SbatchMan || exit
./initEnv.sh
cd .. || exit
source SbatchMan/sourceFile.sh
# 1 node, 1 CPU, 1 GPU, no MPI
rm -rf "$SbM_SOUT"
rm -rf "$SbM_METADATA_HOME"
SbatchMan/newExperiment.sh -p "edu-short" -t 00:05:00 -e "NCU_COO_SPMV" -n 1 -c 1 -g 1 -d 1 -w edu02 -b sudo

source SbatchMan/submit.sh
my_hostname=$(${SbM_UTILS}/hostname.sh)

for file in "$MTX_PATH"/*.sbmtx; do
    echo "----- Testing '$(basename "${file%.*}")' graph -----"
    echo "JOB ID: ${job_id}"

	SbM_submit_function --verbose --expname "$NCU_COO_SPMV_$(basename $file)" --binary sudo $(which ncu) --force-overwrite --set full --export profilation/"profile-$(basename $file)" $BIN -f $file
    echo "JOB ID: ${job_id}"
done
