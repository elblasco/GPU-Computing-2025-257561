# MISC
SRC_DIR=src
BUILD_DIR=build

# DISTRIBUTED_MMIO OPT
CC=g++
DIST_MMIO=distributed_mmio
DIST_MMIO_INCLUDE=$(DIST_MMIO)/include
DIST_MMIO_OBJ=$(BUILD_DIR)/mmio.o $(BUILD_DIR)/mmio_utils.o
DIST_MMIO_OBJ_C=$(DIST_MMIO_OBJ) $(BUILD_DIR)/mmio_c.o
DIST_MMIO_FLAGS=-O3 -g

# CUDA OPT
CUDAC=nvcc
CUDAC_FLAGS=--gpu-architecture=sm_80 -m64 -O3 -g --compiler-options "-Wall -Wextra" -Iinclude -lcusparse -lineinfo
MAIN_SRC=$(SRC_DIR)/main.cu

# MISC
all: clean gpu converter

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

gpu: $(BUILD_DIR)/coo-mul

converter: $(BUILD_DIR)/mtx_to_sbmtx

# DISTRIBUTED_MMIO OPT
$(BUILD_DIR)/mtx_to_sbmtx: distributed_mmio/src/mtx_to_sbmtx.cpp $(DIST_MMIO_OBJ)
	$(CC) $(DIST_MMIO_FLAGS) $(DIST_MMIO_OBJ) -o $@ $<

$(BUILD_DIR)/mmio.o: distributed_mmio/src/mmio.cpp $(BUILD_DIR)
	$(CC) $(DIST_MMIO_FLAGS) -c $< -o $@

$(BUILD_DIR)/mmio_c.o: distributed_mmio/src/mmio_c_wrapper.cpp $(BUILD_DIR)
	$(CC) $(DIST_MMIO_FLAGS) -c $< -o $@ -I$(DIST_MMIO_INCLUDE)

$(BUILD_DIR)/mmio_utils.o: distributed_mmio/src/mmio_utils.cpp $(BUILD_DIR)
	$(CC) $(DIST_MMIO_FLAGS) -c $< -o $@

# CUDA
$(BUILD_DIR)/coo-mul: $(DIST_MMIO_OBJ_C) src/main.cu
	$(CUDAC) $(CUDAC_FLAGS) $^ -o $@
