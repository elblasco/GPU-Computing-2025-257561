STD:=-std=c++17
OUTPUT:=-o build/output.exec
OPT:=-O3
FLAGS:=-Wall $(STD) $(OPT) $(OUTPUT)
GFLAGS:=--gpu-architecture=sm_80 -g -m64 $(OUTPUT) $(STD)
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
