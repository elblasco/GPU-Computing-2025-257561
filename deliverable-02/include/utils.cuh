#pragma once

#include "colours.h"
#include <math.h>
#include <cstddef>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define NUM_TYPE float
#define IDX_TYPE uint32_t
#define MAX_BLOCK_SIZE 128
#define MAX_GRID_SIZE 256
#define NUM_TEST 5
#define WARM_UP_RUN 5
#define OPS_PER_NUN 2
#define MEMEORY_RW 5

#define CEILING(x,y) (((x) + (y) - 1) / (y))

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%u\n", __FILE__, __LINE__); \
        exit(1); \
    }

#define CUDA_MANGED_MALLOC(d_name, type, size)							\
  type * d_name;														\
  CHECK_CUDA(cudaMallocManaged(& d_name, size * sizeof(type)));			\

#define CUDA_FREE(d_name)		  \
  CHECK_CUDA(cudaFree(d_name)); \

#define CUDA_TIMER_DEF(name) \
    cudaEvent_t __timer_start_##name, __timer_stop_##name; \
    CHECK_CUDA(cudaEventCreate(&__timer_start_##name)); \
    CHECK_CUDA(cudaEventCreate(&__timer_stop_##name));

#define CUDA_TIMER_START(name) \
    CHECK_CUDA(cudaEventRecord(__timer_start_##name, 0));

#define CUDA_TIMER_STOP(name) \
    CHECK_CUDA(cudaEventRecord(__timer_stop_##name, 0)); \
    CHECK_CUDA(cudaEventSynchronize(__timer_stop_##name));

#define CUDA_TIMER_DESTROY(name) \
    CHECK_CUDA(cudaEventDestroy(__timer_start_##name)); \
    CHECK_CUDA(cudaEventDestroy(__timer_stop_##name));

#define CUDA_TIMER_ELAPSED(name) \
    ({ float elapsed_##name = 0.0f; \
       CHECK_CUDA(cudaEventElapsedTime(&elapsed_##name, __timer_start_##name, __timer_stop_##name)); \
       elapsed_##name; })

#define CUDA_TIMER_PRINT(name) \
    printf(BRIGHT_CYAN "Timer [%s] elapsed: %f ms\n" RESET, #name, CUDA_TIMER_ELAPSED(name));

#define CUDA_TIMER_INIT(name) CUDA_TIMER_DEF(name) CUDA_TIMER_START(name)

#define CUDA_TIMER_CLOSE(name) CUDA_TIMER_STOP(name) CUDA_TIMER_PRINT(name) CUDA_TIMER_DESTROY(name)

#define CPU_TIMER_DEF(name) \
  std::chrono::high_resolution_clock::time_point __timer_start_##name, __timer_stop_##name;

#define CPU_TIMER_START(name) \
  __timer_start_##name = std::chrono::high_resolution_clock::now();

#define CPU_TIMER_STOP(name) \
  __timer_stop_##name = std::chrono::high_resolution_clock::now();

#define CPU_TIMER_ELAPSED(name) \
  (std::chrono::duration<float>(__timer_stop_##name - __timer_start_##name).count()*1e3)

#define CPU_TIMER_PRINT(name) \
  printf(BRIGHT_CYAN "Timer [%s] elapsed: %f ms\n" RESET, #name, CPU_TIMER_ELAPSED(name));

#define CPU_TIMER_INIT(name) CPU_TIMER_DEF(name) CPU_TIMER_START(name)

#define CPU_TIMER_CLOSE(name) CPU_TIMER_STOP(name) CPU_TIMER_PRINT(name)

typedef void (*gpu_kernel)(const IDX_TYPE *, const IDX_TYPE *, const NUM_TYPE *,
                           const NUM_TYPE *, NUM_TYPE *, const size_t, const size_t);

double geometric_mean(float arr[], size_t n){
    float product = 1;
    for (size_t i = 0; i < n; i++){
	  product = product * arr[i];
	}
    return pow(product, (float)1 / n);
}

float sigma_fn(float* v, float mu, size_t n){
	float sum = 0;
	for (size_t i = 0; i<n; ++i){
		sum += pow(v[i] - mu,2);
	}
	return sqrt(sum / n);
}

double flops_counter(size_t nnz, float ms) {
  size_t flops = OPS_PER_NUN * nnz;
  return (flops / (ms / 1.e3)) / 1.e9;
}
