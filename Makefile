OUTPUT:=-o build/output.exec
FLAGS:=--gpu-architecture=sm_80 -m64 -std=c++17 -O0 -g
CC:=nvcc
OBJS= build/gpu.o build/utils.o build/cpu.o build/tester.o

all: clean build library gpu

gpu: src/main.cu
	$(CC) $(FLAGS) $(OUTPUT) $(OBJS) $^

library:
	$(CC) $(FLAGS) -o build/utils.o -c src/utils.cu
	$(CC) $(FLAGS) -o build/cpu.o -c src/cpu.cpp
	$(CC) $(FLAGS) -o build/gpu.o -c src/gpu.cu
	$(CC) $(FLAGS) -o build/tester.o -c src/tester.cu

clean:
	rm -rf build

build:
	mkdir build	
