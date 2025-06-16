## !! PLEASE MODIFY THE VALUE OF "GROUP_NAME"
## Write here the name of your group
## Please use the name you submitted in the registration form
## !!

RED='\033[0;31m'
PUR='\033[0;35m'
GRE='\033[0;32m'
NC='\033[0m' # No Color

git submodule update --init --recursive

# Create the dataset directory if it doesn't exist
mkdir -p dataset

# Change into the dataset directory
cd dataset

# List of URLs to download
urls=(
  "https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/bone010.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/Dziekonski/dielFilterV3real.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020330.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt160.tar.gz"
)

# Download and extract each file
for url in "${urls[@]}"; do
  file=$(basename "$url")
  dir="${file%.tar.gz}"
  if ! [ -f "$dir.mtx" ]; then
	  
	  echo "Downloading $file..."
	  wget "$url"
	  
	  echo "Extracting $file..."
	  tar -xzf "$file"
	  
	  echo "Moving matrix file from $dir..."
	  if [ -f "$dir/$dir.mtx" ]; then
		  mv "$dir/$dir.mtx" .
	  else
		  echo "Warning: $dir/$dir.mtx not found"
	  fi
	  
	  rm -r "$dir" "$dir.tar.gaz"
  fi
done

cd ..

export BIN=bin/coo-mul
export ITERATIONS=15
export HOST="baldo"
if [[ ! -z $1 ]]; then
    export MTX_PATH=$1
else
    export MTX_PATH="./datasets"
fi
