# Deliverable 1
## Prerequisites
Both the run script anmd the make file works onli on the cluster because uses the `module` package manager.

Please create a directory dataset and download (and extract) into it the follwoing matrices:
+ https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/bone010.tar.gz
+ https://suitesparse-collection-website.herokuapp.com/MM/Dziekonski/dielFilterV3real.tar.gz
+ https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020330.tar.gz
+ https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt160.tar.gz
## To Build It
```
git clone https://github.com/elblasco/GPU-Computing-2025-257561.git
cd GPU-Computing-2025-257561
module load CUDA
make all
```
## To Run It
```
git clone https://github.com/elblasco/GPU-Computing-2025-257561.git
cd GPU-Computing-2025-257561
./run.sh
```
