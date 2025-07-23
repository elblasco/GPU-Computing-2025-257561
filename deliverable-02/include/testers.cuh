#pragma once

#include "colours.h"
#include "gpu.cuh"
#include "utils.cuh"
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <nvtx3/nvToolsExt.h>

void test_spmv(const COO_local<IDX_TYPE, NUM_TYPE> *sparse_matrix,
               kernel_type kernel) {
  float times[NUM_TEST];
  float flops[NUM_TEST];
  float bandwidth[NUM_TEST];
  size_t nnz = sparse_matrix->nnz;

  nvtxRangePushA("Allocating and populating COO arrays in unified memory");

  IDX_TYPE *d_cols, *d_rows;
  NUM_TYPE *d_vals;

  CUDA_MANGED_MALLOC(d_vals, NUM_TYPE, nnz);
  CUDA_MANGED_MALLOC(d_rows, IDX_TYPE, nnz);
  CUDA_MANGED_MALLOC(d_cols, IDX_TYPE, nnz);

  populate_d_arrays(sparse_matrix, d_vals, d_rows, d_cols);

  nvtxRangePop();

  CUDA_TIMER_DEF(gpu_time);

  const size_t grid_size = MIN(MAX_GRID_SIZE, CEILING(nnz, MAX_BLOCK_SIZE));
  const size_t block_size = MIN(MAX_BLOCK_SIZE, nnz);
  const size_t portion = CEILING(nnz, (block_size * grid_size));
  // const size_t cell_per_block = portion * block_size;
  const size_t n_threads = grid_size * block_size;
  // const size_t read_per_block = block_size * ELEM_PER_THREAD;

  for (size_t run = 0; run < NUM_TEST + WARM_UP_RUN; run++) {
    nvtxRangePushA("Allocating and populating arrays for SpMV");

    NUM_TYPE *d_res_array, *d_dense_array;

    CUDA_MANGED_MALLOC(d_res_array, NUM_TYPE, sparse_matrix->nrows);
    CUDA_MANGED_MALLOC(d_dense_array, NUM_TYPE, sparse_matrix->ncols);

    for (size_t j = 0; j < MAX(sparse_matrix->nrows, sparse_matrix->ncols);
         ++j) {
      if (j < sparse_matrix->ncols) {
        d_res_array[j] = 0.0;
      }
      if (j < sparse_matrix->nrows) {
        d_dense_array[j] = 1.0;
      }
    }

    nvtxRangePop();

    switch (kernel) {
    case BASELINE:
      CUDA_TIMER_START(gpu_time);
      nvtxRangePushA("Baseline kernel");
      spmv_with_striding<<<grid_size, block_size>>>(d_rows, d_cols, d_vals,
                                                    d_dense_array, d_res_array,
                                                    nnz, portion, n_threads);
      cudaDeviceSynchronize();
      nvtxRangePop();
      CUDA_TIMER_STOP(gpu_time);
      break;

    case WARP_SHFL:
      CUDA_TIMER_START(gpu_time);
      nvtxRangePushA("Basic WARP kernel");
      spmv_coo_shfl<<<grid_size, block_size>>>(
          d_rows, d_cols, d_vals, d_dense_array, d_res_array, nnz, n_threads);
      cudaDeviceSynchronize();
      nvtxRangePop();
      CUDA_TIMER_STOP(gpu_time);
      break;

    case WARP_SHFL_UNROLL:
      CUDA_TIMER_START(gpu_time);
      nvtxRangePushA("WARP and loop unroll kernel");
      spmv_coo_shfl_unroll<<<grid_size, block_size>>>(
          d_rows, d_cols, d_vals, d_dense_array, d_res_array, nnz, n_threads);
      cudaDeviceSynchronize();
      nvtxRangePop();
      CUDA_TIMER_STOP(gpu_time);
      break;
    case SHARED_MEMORY_SUM:
      CUDA_TIMER_START(gpu_time);
      nvtxRangePushA("Shared memory  kernel");
      shared_prefix_sum<<<grid_size, block_size,
                          block_size * sizeof(NUM_TYPE)>>>(
          d_rows, d_cols, d_vals, d_dense_array, d_res_array, nnz, portion,
          n_threads);
      cudaDeviceSynchronize();
      nvtxRangePop();
      CUDA_TIMER_STOP(gpu_time);
      break;
    }

    float milliseconds = CUDA_TIMER_ELAPSED(gpu_time);

    if (WARM_UP_RUN <= run) {
      flops[run - WARM_UP_RUN] = flops_counter(kernel, nnz, milliseconds);
      times[run - WARM_UP_RUN] = milliseconds;
      bandwidth[run - WARM_UP_RUN] =
          bandwidth_counter(kernel, nnz, milliseconds);
    }

    CUDA_FREE(d_dense_array);
    CUDA_FREE(d_res_array);
  }

  compute_results(times, flops, bandwidth);

  CUDA_TIMER_DESTROY(gpu_time);
  CUDA_FREE(d_vals);
  CUDA_FREE(d_rows);
  CUDA_FREE(d_cols);
}
