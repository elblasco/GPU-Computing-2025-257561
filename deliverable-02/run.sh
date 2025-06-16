#!/usr/bin/env bash

# Always run this script from the root of the repo

source env.sh

# Setup environment
if [[ -z $SbM_HOME ]]; then
    echo -e "${GRE}Setting up experiments environment...${NC}"
    if [[ ! -f SbatchMan/sourceFile.sh ]]; then
        echo -e "${GRE}Initializing SLURM configurations...${NC}"
        cd SbatchMan
        ./initEnv.sh
        cd ..
        source SbatchMan/sourceFile.sh
        # 1 node, 1 CPU, 1 GPU, no MPI
        SbatchMan/newExperiment.sh -p "edu-short" -t 00:04:00 -e BFS_smallD -n 1 -c 1 -g 1 -d 1 -w edu01 -b $BIN # -a hackaton  
        SbatchMan/newExperiment.sh -p "edu-short" -t 00:04:00 -e BFS_largeD -n 1 -c 1 -g 1 -d 1 -w edu01 -b $BIN # -a hackaton  
        SbatchMan/newExperiment.sh -p "edu-short" -t 00:04:00 -e my_matrices -n 1 -c 1 -g 1 -d 1 -w edu01 -b $BIN # -a hackaton 
    else
        source SbatchMan/sourceFile.sh
    fi
fi

echo -e "${GRE}Building $BIN...${NC}"
make clean $BIN
source SbatchMan/submit.sh
my_hostname=$(${SbM_UTILS}/hostname.sh)

#echo -e "${GRE}%% Running tests on small-diameter graphs %%${NC}"
for file in "$MTX_PATH"/*.mtx; do
	echo "$file"
    echo "----- Testing '$(basename "${file%.*}")' graph -----"
    SbM_submit_function --verbose --expname $expname --binary $BIN -f "$MTX_PATH/$file" -n $ITERATIONS
    echo "JOB ID: ${job_id}"
done

# echo -e "${GRE}%% Running tests on large-diameter graphs %%${NC}"
# for gi in ${!GRAPHS_LARGE_D[@]}; do
#     graph=${GRAPHS_LARGE_D[$gi]}
#     echo "----- Testing '$(basename "${graph%.*}")' graph -----"
#     SbM_submit_function --verbose --expname $expname --binary $BIN -f "$MTX_PATH/$graph" -n $ITERATIONS
#     echo "JOB ID: ${job_id}"
# done

# echo -e "${GRE}%% Running tests on Graph500 graphs %%${NC}"
# for gi in ${!MY_MATRICES[@]}; do
#     graph=${MY_MATRICES[$gi]}
#     echo "----- Testing '$(basename "${graph%.*}")' graph -----"
#     SbM_submit_function --verbose --expname $expname --binary $BIN -f "$MTX_PATH/$graph" -n $ITERATIONS
#     echo "JOB ID: ${job_id}"
# done
