STD:=-std=c++17
LIBRARY:=-I./library
OUTPUT:=-o build/output.exec
OPT:=-O3
FLAGS:=-Wall $(STD) $(LIBRARY) $(OPT) $(OUTPUT)

all: clean build

build: ./cpu/main.cpp
	mkdir build
	g++ $(FLAGS) $^

clean:
	rm -rf build
