#pragma once

#include "colours.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_TYPE float
#define IDX_TYPE size_t
#define MAX_BLOCK_SIZE 128
#define MAX_GRID_SIZE 256
#define NUM_TEST 5
#define WARM_UP_RUN 5
#define WARP_SIZE 32
#define BANKS_NUM 32
#define FULL_WARP_MASK 0xffffffff
#define OPS_PER_NUN_BASELINE 2
#define MEMEORY_RW_BASELINE 5
#define OPS_PER_NUN_WARP_SHFL 6
#define MEMEORY_RW_WARP_SHFL 5
#define OPS_PER_NUN_WARP_SHFL_UNROLL 6
#define MEMEORY_RW_WARP_SHFL_UNROLL 5
#define ELEM_PER_THREAD 2
#define LOG_WARP 5
#define BANKS_NUM_LOG 5
#define CONFLICT_FREE_OFFSET(n)((n) >> BANKS_NUM + (n) >> (2 * BANKS_NUM_LOG))

#define CEILING(X, Y) (((X) + (Y) - 1) / (Y))
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define CONFLICT_FREE_OFFSET(n) ((n) >> BANKS_NUM + (n) >> (2 * BANKS_NUM_LOG))

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, RED "CUDA error: %s\n" RESET,                            \
              cudaGetErrorString(status));                                     \
      fflush(stderr);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      fprintf(stderr, RED "cuSPARSE error at %d: %d\n" RESET, __LINE__,        \
              status);                                                         \
      fflush(stderr);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

#define CUDA_MANGED_MALLOC(d_name, type, size)                                 \
  type *d_name;                                                                \
  CHECK_CUDA(cudaMallocManaged(&d_name, size * sizeof(type)));

#define CUDA_FREE(d_name) CHECK_CUDA(cudaFree(d_name));

#define CUDA_TIMER_DEF(name)                                                   \
  cudaEvent_t __timer_start_##name, __timer_stop_##name;                       \
  CHECK_CUDA(cudaEventCreate(&__timer_start_##name));                          \
  CHECK_CUDA(cudaEventCreate(&__timer_stop_##name));

#define CUDA_TIMER_START(name)                                                 \
  CHECK_CUDA(cudaEventRecord(__timer_start_##name, 0));

#define CUDA_TIMER_STOP(name)                                                  \
  CHECK_CUDA(cudaEventRecord(__timer_stop_##name, 0));                         \
  CHECK_CUDA(cudaEventSynchronize(__timer_stop_##name));

#define CUDA_TIMER_DESTROY(name)                                               \
  CHECK_CUDA(cudaEventDestroy(__timer_start_##name));                          \
  CHECK_CUDA(cudaEventDestroy(__timer_stop_##name));

#define CUDA_TIMER_ELAPSED(name)                                               \
  ({                                                                           \
    float elapsed_##name = 0.0f;                                               \
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_##name, __timer_start_##name,     \
                                    __timer_stop_##name));                     \
    elapsed_##name;                                                            \
  })

#define CUDA_TIMER_PRINT(name)                                                 \
  printf(BRIGHT_CYAN "Timer [%s] elapsed: %f ms\n" RESET, #name,               \
         CUDA_TIMER_ELAPSED(name));

#define CUDA_TIMER_INIT(name) CUDA_TIMER_DEF(name) CUDA_TIMER_START(name)

#define CUDA_TIMER_CLOSE(name)                                                 \
  CUDA_TIMER_STOP(name) CUDA_TIMER_PRINT(name) CUDA_TIMER_DESTROY(name)

enum kernel_type {
  BASELINE,
  WARP_SHFL,
  WARP_SHFL_UNROLL,
  SHARED_MEMORY_SUM,
  SHARED_MEMORY_BANK,
  CUSPARSE
};

double geometric_mean(const float *arr, size_t n) {
  float product = 1;
  for (size_t i = 0; i < n; i++) {
    product = product * arr[i];
  }
  return pow(product, (float)1 / n);
}

float sigma_fn(const float *v, float mu, size_t n) {
  float sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += pow(v[i] - mu, 2);
  }
  return sqrt(sum / n);
}

double flops_counter(kernel_type kernel, size_t nnz, float ms) {
  size_t flops = 0;
  switch (kernel) {
  case BASELINE:
    flops = OPS_PER_NUN_BASELINE * nnz;
    break;
  case WARP_SHFL:
  case WARP_SHFL_UNROLL:
    flops = nnz * (2 + LOG_WARP);
    break;
  }
  return (flops / (ms / 1.e3)) / 1.e9;
}

float bandwidth_counter(const kernel_type kernel, const size_t nnz,
                        const float milliseconds) {
  float total_memory_rw = 0;
  switch (kernel) {
  case BASELINE:
    total_memory_rw = nnz * MEMEORY_RW_BASELINE;
    break;
  case WARP_SHFL:
    total_memory_rw = nnz * MEMEORY_RW_WARP_SHFL;
    break;
  case WARP_SHFL_UNROLL:
    total_memory_rw = nnz * MEMEORY_RW_WARP_SHFL_UNROLL;
    break;
  }
  return (sizeof(NUM_TYPE) * total_memory_rw / milliseconds) / 1e12;
}

// float bandwidth_counter(const kernel_type kernel, const size_t nnz,
//                         const float milliseconds) {
//   return (nnz * sizeof(NUM_TYPE) * memory_r_w(kernel) / milliseconds) / 1e12;
// }

void compute_results(const float *times, const float *flops,
                     const float *bandwidth) {
  double times_mu = geometric_mean(times, NUM_TEST);
  double times_sigma = sigma_fn(times, times_mu, NUM_TEST);

  printf("This kernel executed with an average of %lf ms with std.dev. of %lf "
         "ms\n",
         times_mu, times_sigma);

  double flops_mu = geometric_mean(flops, NUM_TEST);
  double flops_sigma = sigma_fn(flops, flops_mu, NUM_TEST);

  printf("This kernel produced an average of %lf GFLOP/s with std.dev. of %lf "
         "GFLOP/s\n",
         flops_mu, flops_sigma);

  double bandwidth_mu = geometric_mean(bandwidth, NUM_TEST);
  double bandwidth_sigma = sigma_fn(bandwidth, bandwidth_mu, NUM_TEST);

  printf("This kernel produced an avergare bandwidth of %lf GB/s with std.dev. "
         "of %lf GB/s the theoretical maximun is 864 GB/s\n",
         bandwidth_mu, bandwidth_sigma);
}

void populate_d_arrays(const COO_local<IDX_TYPE, NUM_TYPE> *sparse_matrix,
                       NUM_TYPE *d_vals, IDX_TYPE *d_rows, IDX_TYPE *d_cols) {
  if (sparse_matrix->val != nullptr) {
    cudaMemset(d_vals, 1, sparse_matrix->nnz * sizeof(NUM_TYPE));
    for (size_t i = 0; i < sparse_matrix->nnz; ++i) {
      d_rows[i] = sparse_matrix->row[i];
      d_cols[i] = sparse_matrix->col[i];
    }
  } else {
    for (size_t i = 0; i < sparse_matrix->nnz; ++i) {
      d_vals[i] = sparse_matrix->val[i];
      d_rows[i] = sparse_matrix->row[i];
      d_cols[i] = sparse_matrix->col[i];
    }
  }
}

void validate_result(const NUM_TYPE *d_res_array, const NUM_TYPE *test_result) {
  for (size_t x = 0; x <= 1000; x++) {
    if (fabs(d_res_array[x] - test_result[x]) > 0.0005) {
      printf("At index %lu result contains %f while the test contains %f\n", x,
             d_res_array[x], test_result[x]);
    } else if (x == 1000) {
      printf(GREEN "Everything is fine" RESET "\n");
    }
  }
}
