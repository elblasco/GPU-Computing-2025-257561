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

NUM_TYPE *quality_assurance(const NUM_TYPE *d_coo_vals, const IDX_TYPE *d_rows,
                            const IDX_TYPE *d_cols,
                            const NUM_TYPE *d_dense_array, const size_t nnz,
                            const size_t nrows, const size_t portion,
                            const size_t grid_size, const size_t block_size) {
  nvtxRangePushA("Allocating a populating validation array");
  CUDA_MANGED_MALLOC(d_res_array, NUM_TYPE, nrows);
  for (size_t i = 0; i < nrows; ++i) {
    d_res_array[i] = 0.0;
  }
  nvtxRangePop();
  spmv_with_striding<<<grid_size, block_size>>>(
      d_rows, d_cols, d_coo_vals, d_dense_array, d_res_array, nnz, portion,
      grid_size * block_size);
  cudaDeviceSynchronize();
  return d_res_array;
}

void cuSPARSE_kernel(const COO_local<IDX_TYPE, NUM_TYPE> *sparse_matrix,
                     const NUM_TYPE *d_dense_array, const IDX_TYPE *d_rows,
                     const IDX_TYPE *d_cols, const NUM_TYPE *d_vals,
                     NUM_TYPE *d_res_array, size_t nnz) {
  NUM_TYPE alpha = 1.0, beta = 1.0;
  cusparseHandle_t handle = NULL;
  printf("Creating the cusparse handler\n");
  fflush(stdout);
  CHECK_CUSPARSE(cusparseCreate(&handle));

  cusparseConstDnVecDescr_t dense = NULL;
  cusparseDnVecDescr_t res = NULL;

  IDX_TYPE ncols = sparse_matrix->ncols;
  IDX_TYPE nrows = sparse_matrix->nrows;
  printf("Creating the const vec\n");
  fflush(stdout);
  cusparseCreateConstDnVec(&dense, ncols, (void *)d_dense_array, CUDA_R_32F);

  printf("Creating the result vec\n");
  fflush(stdout);
  cusparseCreateDnVec(&res, nrows, d_res_array, CUDA_R_32F);

  printf("Creating the const matrix\n");
  fflush(stdout);
  cusparseConstSpMatDescr_t spMatDescr = NULL;
  cusparseCreateConstCoo(&spMatDescr, nrows, ncols, nnz, d_rows, d_cols, d_vals,
                         CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO,
                         CUDA_R_32F);

  size_t buffer_size = 0;

  printf("Computing the buffer size\n");
  fflush(stdout);
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          spMatDescr, dense, &beta, res, CUDA_R_32F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);

  printf("The buffer size id %lu\n", buffer_size);
  fflush(stdout);

  void *buffer = NULL;
  CHECK_CUDA(cudaMalloc(&buffer, buffer_size));

  printf("Preprocess the matrix\n");
  fflush(stdout);
  nvtxRangePushA("Preprocess  cuSPARSE");
  cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          spMatDescr, dense, &beta, res, CUDA_R_32F,
                          CUSPARSE_SPMV_ALG_DEFAULT, buffer);
  nvtxRangePop();

  printf("Doing the multiplication\n");
  fflush(stdout);

  nvtxRangePushA("Multiplication  cuSPARSE");
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spMatDescr,
               dense, &beta, res, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
               buffer);
  nvtxRangePop();

  printf("Finished the multiplication\n");
  fflush(stdout);

  if (buffer_size > 0) {
    printf("Freeing the buffer memory\n");
    fflush(stdout);
    CUDA_FREE(buffer);
  }

  printf("Freeing the matrix\n");
  fflush(stdout);
  CHECK_CUSPARSE(cusparseDestroySpMat(spMatDescr));

  printf("Freeing the const dense vector\n");
  fflush(stdout);
  CHECK_CUSPARSE(cusparseDestroyDnVec(dense));

  printf("Freeing the result vector\n");
  fflush(stdout);
  CHECK_CUSPARSE(cusparseDestroyDnVec(res));

  CHECK_CUSPARSE(cusparseDestroy(handle));
}

void test_spmv(const COO_local<IDX_TYPE, NUM_TYPE> *sparse_matrix,
               kernel_type kernel) {
  float times[NUM_TEST];
  float flops[NUM_TEST];
  float bandwidth[NUM_TEST];
  size_t nnz = sparse_matrix->nnz;

  nvtxRangePushA("Allocating and populating COO arrays in unified memory");

  CUDA_MANGED_MALLOC(d_vals, NUM_TYPE, nnz);
  CUDA_MANGED_MALLOC(d_rows, IDX_TYPE, nnz);
  CUDA_MANGED_MALLOC(d_cols, IDX_TYPE, nnz);

  populate_d_arrays(sparse_matrix, d_vals, d_rows, d_cols);

  nvtxRangePop();

  CUDA_TIMER_DEF(gpu_time);

  const size_t grid_size = MIN(MAX_GRID_SIZE, CEILING(nnz, MAX_BLOCK_SIZE));
  const size_t block_size = MIN(MAX_BLOCK_SIZE, nnz);
  const size_t portion = CEILING(nnz, (block_size * grid_size));
  const size_t cell_per_block = portion * block_size;
  const size_t n_threads = grid_size * block_size;
  const size_t read_per_block = block_size * ELEM_PER_THREAD;

  for (size_t run = 0; run < NUM_TEST + WARM_UP_RUN; run++) {
	nvtxRangePushA("Allocating and populating arrays for SpMV");

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
      spmv_coo_shfl<<<grid_size, block_size>>>(d_rows, d_cols, d_vals,
                                               d_dense_array, d_res_array, nnz,
                                               cell_per_block, n_threads);
      cudaDeviceSynchronize();
	  nvtxRangePop();
      CUDA_TIMER_STOP(gpu_time);
      break;

    case WARP_SHFL_UNROLL:
      CUDA_TIMER_START(gpu_time);
	  nvtxRangePushA("WARP and loop unroll kernel");
      spmv_coo_shfl_unroll<<<grid_size, block_size>>>(
          d_rows, d_cols, d_vals, d_dense_array, d_res_array, nnz,
          cell_per_block, n_threads);
      cudaDeviceSynchronize();
	  nvtxRangePop();
      CUDA_TIMER_STOP(gpu_time);
      break;
    case SHARED_MEMORY_SUM:
      // CUDA_TIMER_START(gpu_time);
	  // nvtxRangePushA("Shared memory  kernel");
      // shared_prefix_sum<<<grid_size, block_size,
      //                     2 * block_size * sizeof(NUM_TYPE)>>>(
	  // 														   d_rows, d_cols, d_vals, d_dense_array, d_res_array, nnz, read_per_block);
      // cudaDeviceSynchronize();
	  // nvtxRangePop();
      // CUDA_TIMER_STOP(gpu_time);
      break;
    case CUSPARSE:
	  nvtxRangePushA("Whole cuSPARSE");
      cuSPARSE_kernel(sparse_matrix, d_dense_array, d_rows, d_cols, d_vals,
                      d_res_array, nnz);
	  nvtxRangePop();
      break;
    }

    float milliseconds = CUDA_TIMER_ELAPSED(gpu_time);

    if (WARM_UP_RUN <= run) {
      flops[run - WARM_UP_RUN] = flops_counter(kernel, nnz, milliseconds);
      times[run - WARM_UP_RUN] = milliseconds;
      bandwidth[run - WARM_UP_RUN] =
          bandwidth_counter(kernel, nnz, milliseconds);

      // NUM_TYPE *test_result = quality_assurance(
      //     d_vals, d_rows, d_cols, d_dense_array, nnz, sparse_matrix->nrows,
      //     portion, grid_size, block_size);

      // validate_result(d_res_array, test_result);

      // CUDA_FREE(test_result);
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
