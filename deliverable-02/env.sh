RED='\033[0;31m'
PUR='\033[0;35m'
GRE='\033[0;32m'
NC='\033[0m' # No Color

export BIN=build/coo-mul
export ITERATIONS=15
export HOST="baldo"
export EXPERIMENT_NAME="COO_SPMV"

ml CUDA

git submodule update --init --recursive

echo -e "${GRE}Building $BIN...${NC}"
make all

if [ $? -ne 0 ]; then
    echo -e "${RED} Build failed exiting${NC}"
	exit
fi

# Create the dataset directory if it doesn't exist
mkdir -p datasets

if [[ -n $1 ]]; then
    export MTX_PATH="$1"
else
    export MTX_PATH="./datasets"
fi

# Change into the dataset directory
cd "$MTX_PATH" || exit

# List of URLs to download
urls=(
  "https://suitesparse-collection-website.herokuapp.com/MM/Dziekonski/dielFilterV3real.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Flan_1565.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020330.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt160.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Queen_4147.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/VLSI/vas_stokes_2M.tar.gz"
)

# Download and extract each file
for url in "${urls[@]}"; do
  file=$(basename "$url")
  dir="${file%.tar.gz}"
  if ! [ -f "$dir.sbmtx" ]; then

	  echo -e "${GRE}Downloading $file...${NC}"
	  wget "$url"

	  echo -e "${PUR}Extracting $file...${NC}"
	  tar -xzf "$file"

	  if [ -f "$dir/$dir.mtx" ]; then
		  echo -e "${PUR}Moving $dir/$dir.mtx to $dir.mtx${NC}"
		  mv "$dir/$dir.mtx" .
		  echo -e "${PUR}Converting $dir.mtx to $dir.sbmtx${NC}"
		  ./../build/mtx_to_sbmtx "$dir.mtx"
		  echo "The last command returned $?"
	  else
		  echo -e "${RED}Warning: $dir/$dir.mtx not found${NC}"
	  fi
  fi
  rm -fr "$dir" "$file"
done

cd ..
