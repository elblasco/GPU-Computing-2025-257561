#pragma once

#include "gpu.cuh"
#include "utils.cuh"
#include "colours.h"
#include <cmath>
#include <csignal>
#include <cstddef>

NUM_TYPE *quality_assurance(const NUM_TYPE *d_coo_vals, const IDX_TYPE *d_rows,
                            const IDX_TYPE *d_cols,
                            const NUM_TYPE *d_dense_array, const size_t nnz,
                            const size_t nrows, const size_t portion,
                            const size_t grid_size, const size_t block_size) {
  CUDA_MANGED_MALLOC(d_res_array, NUM_TYPE, nrows);
  spmv_with_striding<<<grid_size, block_size>>>(
      d_rows, d_cols, d_coo_vals, d_dense_array, d_res_array, nnz, portion);
  return d_res_array;
}

void test_spmv(const COO_local<IDX_TYPE, NUM_TYPE> *sparse_matrix,
               kernel_type kernel) {
  float times[NUM_TEST];
  float flops[NUM_TEST];
  float bandwidth[NUM_TEST];
  size_t nnz = sparse_matrix->nnz;

  CUDA_MANGED_MALLOC(d_vals, NUM_TYPE, nnz);
  CUDA_MANGED_MALLOC(d_rows, IDX_TYPE, nnz);
  CUDA_MANGED_MALLOC(d_cols, IDX_TYPE, nnz);

  populate_d_arrays(sparse_matrix, d_vals, d_rows, d_cols);

  CUDA_TIMER_DEF(gpu_time);

  size_t grid_size = MIN(MAX_GRID_SIZE, CEILING(nnz, MAX_BLOCK_SIZE));
  size_t block_size = MIN(MAX_BLOCK_SIZE, nnz);
  size_t portion = CEILING(nnz, (block_size * grid_size));

  for (size_t run = 0; run < NUM_TEST + WARM_UP_RUN; run++) {
    CUDA_MANGED_MALLOC(d_res_array, NUM_TYPE, sparse_matrix->nrows);
    CUDA_MANGED_MALLOC(d_dense_array, NUM_TYPE, sparse_matrix->ncols);

    for (size_t j = 0; j < MAX(sparse_matrix->nrows, sparse_matrix->ncols);
         ++j) {
      if (j < sparse_matrix->ncols) {
        d_res_array[j] = 0.0;
      }
      if (j < sparse_matrix->nrows) {
        d_dense_array[j] = 0.0;
      }
    }

    switch (kernel) {
    case BASELINE:
      // printf("Kernel ASSIGNEMENT_01 will be executed on %lu threads, each of
      // them should cover at most %lu elements\n", (grid_size * block_size),
      // portion);
      CUDA_TIMER_START(gpu_time);
      spmv_with_striding<<<grid_size, block_size>>>(
          d_rows, d_cols, d_vals, d_dense_array, d_res_array, nnz, portion);
      cudaDeviceSynchronize();
      CUDA_TIMER_STOP(gpu_time);
      break;

    case WARP_SHFL:
      // printf("Kernel WARP_SHFL will be executed on %lu threads\n", (grid_size
      // * block_size));
      CUDA_TIMER_START(gpu_time);
      spmv_coo_shfl<<<grid_size, block_size>>>(d_rows, d_cols, d_vals,
                                               d_dense_array, d_res_array, nnz);
      cudaDeviceSynchronize();
      CUDA_TIMER_STOP(gpu_time);
      break;

    case WARP_SHFL_UNROLL:
      // printf("Kernel WARP_SHFL_UNROLL will be executed on %lu threads\n",
      // (grid_size * block_size));
      CUDA_TIMER_START(gpu_time);
      spmv_coo_shfl_unroll<<<grid_size, block_size>>>(
          d_rows, d_cols, d_vals, d_dense_array, d_res_array, nnz);
      cudaDeviceSynchronize();
      CUDA_TIMER_STOP(gpu_time);
      break;
    }

    float milliseconds = CUDA_TIMER_ELAPSED(gpu_time);

    if (WARM_UP_RUN <= run) {
      flops[run - WARM_UP_RUN] = flops_counter(kernel, nnz, milliseconds);
      times[run - WARM_UP_RUN] = milliseconds;
      bandwidth[run - WARM_UP_RUN] =
          bandwidth_counter(kernel, nnz, milliseconds);

      NUM_TYPE *test_result = quality_assurance(
          d_vals, d_rows, d_cols, d_dense_array, nnz, sparse_matrix->nrows,
          portion, grid_size, block_size);

      for (size_t x = 0; x < sparse_matrix->nrows; x++) {
        if (fabs(d_res_array[x] - test_result[x]) > 0.0005) {
          printf("At index %lu result contains %f\n", x, d_res_array[x]);
        }
		if(x == 1000){
		  printf(GREEN "Everything is fine" RESET "\n");
		}
      }
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
