#pragma once

#include <cstddef>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define MAX_BLOCK_SIZE 1024
#define MAX_GRID_SIZE 256
#define IDX_TYPE size_t
#define NUM_TYPE double
#define NUM_TEST 10
#define OPS_PER_NUN 2
#define MEMEORY_RW 5
#define cudaCheckError(ans)                                                    \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }

typedef void (*gpu_kernel)(const IDX_TYPE *, const IDX_TYPE *, const NUM_TYPE *,
                           const NUM_TYPE *, NUM_TYPE *, const size_t, const size_t);

inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}
