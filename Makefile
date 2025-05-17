STD:=-std=c++17
GPU-OUTPUT:=-o build/gpu-output.exec
CPU-OUTPUT:=-o build/cpu-output.exec
OPT:=-O0 -g
FLAGS:=-Wall $(STD) $(OPT) $(CPU-OUTPUT)
GFLAGS:=--gpu-architecture=sm_80 -m64 $(STD) $(GPU-OUTPUT)
gpu: clean gpu-build

gpu-build: ./src/gpu/main.cu
	mkdir build
	nvcc $(GFLAGS) $^

cpu: clean cpu-build

cpu-build: ./src/cpu/main.cpp
	mkdir build
	g++ $(FLAGS) $^

clean:
	rm -rf build
