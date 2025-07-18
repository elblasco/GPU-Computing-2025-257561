#pragma once

#include "colours.h"
#include "gpu.cuh"
#include "utils.cuh"
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cusparse.h>

NUM_TYPE *quality_assurance(const NUM_TYPE *d_coo_vals, const IDX_TYPE *d_rows,
                            const IDX_TYPE *d_cols,
                            const NUM_TYPE *d_dense_array, const size_t nnz,
                            const size_t nrows, const size_t portion,
                            const size_t grid_size, const size_t block_size) {
  CUDA_MANGED_MALLOC(d_res_array, NUM_TYPE, nrows);
  for (size_t i = 0; i < nrows; ++i) {
    d_res_array[i] = 0.0;
  }
  spmv_with_striding<<<grid_size, block_size>>>(
      d_rows, d_cols, d_coo_vals, d_dense_array, d_res_array, nnz, portion);
  cudaDeviceSynchronize();
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
        d_dense_array[j] = 1.0;
      }
    }

    switch (kernel) {
    case BASELINE:
      CUDA_TIMER_START(gpu_time);
      spmv_with_striding<<<grid_size, block_size>>>(
          d_rows, d_cols, d_vals, d_dense_array, d_res_array, nnz, portion);
      cudaDeviceSynchronize();
      CUDA_TIMER_STOP(gpu_time);
      break;

    case WARP_SHFL:
      CUDA_TIMER_START(gpu_time);
      spmv_coo_shfl<<<grid_size, block_size>>>(d_rows, d_cols, d_vals,
                                               d_dense_array, d_res_array, nnz);
      cudaDeviceSynchronize();
      CUDA_TIMER_STOP(gpu_time);
      break;

    case WARP_SHFL_UNROLL:
      CUDA_TIMER_START(gpu_time);
      spmv_coo_shfl_unroll<<<grid_size, block_size>>>(
          d_rows, d_cols, d_vals, d_dense_array, d_res_array, nnz);
      cudaDeviceSynchronize();
      CUDA_TIMER_STOP(gpu_time);
      break;
    case SHARED_MEMORY_SUM:
      CUDA_TIMER_START(gpu_time);
      shared_prefix_sum<<<grid_size, block_size,
                          2 * block_size * sizeof(NUM_TYPE)>>>(
          d_rows, d_cols, d_vals, d_dense_array, d_res_array, nnz);
      cudaDeviceSynchronize();
      CUDA_TIMER_STOP(gpu_time);
      break;
    case CUSPARSE:
      cusparseHandle_t *handle = NULL;
      cusparseCreate(handle);

      cusparseConstDnVecDescr_t dense = NULL;
      cusparseDnVecDescr_t res = NULL;
      cusparseConstDnVecGet(dense, &sparse_matrix->ncols, d_dense_array,
                            CUDA_R_32F);

      cusparseDnVecGet(res, sparse_matrix->nrows, d_res_array, CUDA_R_32F);

      cusparseConstSpMatDescr_t *spMatDescr = NULL;
      cusparseCreateConstCoo(spMatDescr, &sparse_matrix->nrows,
                             sparse_matrix->ncols, sparse_matrix->nnz, d_rows,
                             d_cols, d_vals, CUSPARSE_INDEX_64I,
                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

      size_t buffer_size = 0;

      cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 1,
                              spMatDescr, dense, 1, res, CUDA_R_32F,
                              CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);

      CUDA_MANGED_MALLOC(buffer, NUM_TYPE, buffer_size);

      cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 1,
                              spMatDescr, dense, 1, res, CUDA_R_32F,
                              CUSPARSE_SPMV_ALG_DEFAULT, buffer);

      cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 1, spMatDescr,
                   dense, 1, res, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                   buffer);

      // cusparseSpMV(cusparseHandle_t          handle,
      //        cusparseOperation_t       opA,
      //        const void*               alpha,
      //        cusparseConstSpMatDescr_t matA,  // non-const descriptor
      //        supported cusparseConstDnVecDescr_t vecX,  // non-const
      //        descriptor supported const void*               beta,
      //        cusparseDnVecDescr_t      vecY,
      //        cudaDataType              computeType,
      //        cusparseSpMVAlg_t         alg,
      //        void*                     externalBuffer)

      CUDA_FREE(buffer);
      cusparseDestroySpMat(spMatDescr);
      cusparseDestroyDnVec(dense);
      cusparseDestroyDnVec(res);
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

      validate_result(d_res_array, test_result);
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
