STD:=-std=c++17
OUTPUT:=-o build/output.exec
OPT:=-O3
FLAGS:=-Wall $(STD) $(OPT) $(OUTPUT)

gpu: clean gpu-build

gpu-build: ./gpu/main.cu
	mkdir build
	nvcc --gpu-architecture=sm_80 $(OUTPUT) $^

cpu: clean cpu-build

cpu-build: ./cpu/main.cpp
	mkdir build
	g++ $(FLAGS) $^

clean:
	rm -rf build
