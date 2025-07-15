#pragma once

#include "colours.h"
#include <math.h>
#include <cstddef>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define NUM_TYPE float
#define IDX_TYPE uint64_t
#define MAX_BLOCK_SIZE 128
#define MAX_GRID_SIZE 256
#define NUM_TEST 5
#define WARM_UP_RUN 5
#define OPS_PER_NUN 2
#define MEMEORY_RW 5

#define CEILING(x,y) (((x) + (y) - 1) / (y))

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

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

enum kernel_type {
  BASELINE,
  WARP_SHFL,
  WARP_SHFL_UNROLL
};

double geometric_mean(const float *arr, size_t n){
    float product = 1;
    for (size_t i = 0; i < n; i++){
	  product = product * arr[i];
	}
    return pow(product, (float)1 / n);
}

float sigma_fn(const float* v, float mu, size_t n){
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

void compute_results(const float* times, const float* flops, const float* bandwidth) {
  double times_mu = geometric_mean(times, NUM_TEST);
  double times_sigma = sigma_fn(times, times_mu, NUM_TEST);

  printf("This kernel executed with an average of %lf ms with std.dev. of %lf ms\n", times_mu, times_sigma);

  double flops_mu = geometric_mean(flops, NUM_TEST);
  double flops_sigma = sigma_fn(flops, flops_mu, NUM_TEST);

  printf("This kernel produced an average of %lf GFLOP/s with std.dev. of %lf GFLOP/s\n", flops_mu, flops_sigma);

  double bandwidth_mu = geometric_mean(bandwidth, NUM_TEST);
  double bandwidth_sigma = sigma_fn(bandwidth, bandwidth_mu, NUM_TEST);

  printf("This kernel produced an avergare bandwidth of %lf GB/s with std.dev. of %lf GB/s the theoretical maximun is 864 GB/s\n", bandwidth_mu, bandwidth_sigma);
}

void populate_d_arrays(const mmio_coo_u64_f32_t *sparse_matrix, NUM_TYPE *d_vals, IDX_TYPE *d_rows, IDX_TYPE *d_cols) {
  // IMPORTANT inversion of column and rows to have a row major COO
  for (size_t i = 0; i < sparse_matrix -> nnz; ++i){
	d_vals[i] = sparse_matrix -> val[i];
	d_rows[i] = sparse_matrix -> col[i];
	d_cols[i] = sparse_matrix -> row[i];
  }
}
