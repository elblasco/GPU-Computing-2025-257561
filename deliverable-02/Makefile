# MISC
SRC_DIR=src
BUILD_DIR=build

# DISTRIBUTED_MMIO OPT
DIST_MMIO=distributed_mmio
DIST_MMIO_INCLUDE=$(DIST_MMIO)/include
DIST_MMIO_OBJ=$(BUILD_DIR)/dmmio_mmio.o $(BUILD_DIR)/dmmio_utils.o
DIST_MMIO_FLAGS=-O3 -g
DIST_MMIO_SRCS=$(filter-out $(DIST_MMIO)/src/mtx_to_bmtx.cpp, $(wildcard $(DIST_MMIO)/src/*))

# CUDA OPT
CUDAC=nvcc
CUDAC_FLAGS=--gpu-architecture=sm_80 -m64 -O3 -g --compiler-options "-Wall -Wextra"
PROF_FLAGS=-lineinfo
MAIN_SRC=$(SRC_DIR)/main.cu

#SRCS=$(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/*.cuh)

# MISC
all: $(BUILD_DIR)/coo-mul converter

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

converter: $(BUILD_DIR)/mtx_to_sbmtx

# DISTRIBUTED_MMIO OPT
$(BUILD_DIR)/mtx_to_sbmtx: distributed_mmio/src/mtx_to_sbmtx.cpp $(DIST_MMIO_OBJ)
	g++ $(DIST_MMIO_FLAGS) $< $(DIST_MMIO_OBJ) -o $@

$(BUILD_DIR)/dmmio_mmio.o: distributed_mmio/src/mmio.cpp $(BUILD_DIR)
	g++ $(DIST_MMIO_FLAGS) -c $< -o $@

$(BUILD_DIR)/dmmio_utils.o: distributed_mmio/src/mmio_utils.cpp $(BUILD_DIR)
	g++ $(DIST_MMIO_FLAGS) -c $< -o $@

# CUDA
$(BUILD_DIR)/coo-mul: $(DIST_MMIO_OBJ) $(MAIN_SRC)
	$(CUDAC) $(CUDAC_FLAGS) $< -o $@
#$(CUDAC) $(CUDAC_FLAGS) -I$(DIST_MMIO_INCLUDE) --compiler-options "$(C_FLAGS)" -o $@ $^ $(DIST_MMIO_SRCS)
