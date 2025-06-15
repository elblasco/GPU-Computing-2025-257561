## !! PLEASE MODIFY THE VALUE OF "GROUP_NAME"
## Write here the name of your group
## Please use the name you submitted in the registration form
## !!

RED='\033[0;31m'
PUR='\033[0;35m'
GRE='\033[0;32m'
NC='\033[0m' # No Color

export BIN=bin/coo-mul
export ITERATIONS=15
export HOST="baldo"
if [[ ! -z $1 ]]; then
    export MTX_PATH=$1
else
    export MTX_PATH="./datasets"
fi

# Read graph paths from matrices_list.txt in each subfolder
GRAPHS_SMALL_D=()
while IFS= read -r line; do
    GRAPHS_SMALL_D+=("$line")
done < "$MTX_PATH/small_diameter/matrices_list.txt"

GRAPHS_LARGE_D=()
while IFS= read -r line; do
    GRAPHS_LARGE_D+=("$line")
done < "$MTX_PATH/large_diameter/matrices_list.txt"

MY_MATRICES=()
while IFS= read -r line; do
    MY_MATRICES+=("$line")
done < "$MTX_PATH/graph500/matrices_list.txt"

ALL_GRAPHS=()
while IFS= read -r line; do
    ALL_GRAPHS+=("$line")
done < "$MTX_PATH/matrices_list.txt"

export GRAPHS_SMALL_D
export GRAPHS_LARGE_D
export MY_MATRICES
export ALL_GRAPHS
