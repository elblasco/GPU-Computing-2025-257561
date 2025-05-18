#include "include/gpu.h"
#include "include/macros.h"

__global__ void spmv_with_striding(const IDX_TYPE *row, const IDX_TYPE *col,
                                   const NUM_TYPE *val, const NUM_TYPE *arr,
                                   NUM_TYPE *res, const size_t nnz,
                                   const size_t portion) {
  size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t n_thread = blockDim.x * gridDim.x;

  IDX_TYPE start = thread_idx;
  IDX_TYPE end = (n_thread * portion) + thread_idx;

  for (IDX_TYPE i = start; i < end && i < nnz; i += n_thread) {
    atomicAdd(&res[row[i]], val[i] * arr[col[i]]);
  }
}

__global__ void spmv_without_striding(const IDX_TYPE *row, const IDX_TYPE *col,
                                      const NUM_TYPE *val, const NUM_TYPE *arr,
                                      NUM_TYPE *res, const size_t nnz,
                                      const size_t portion) {
  size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  
  IDX_TYPE start = portion * thread_idx;
  IDX_TYPE end = start + portion;

  for (IDX_TYPE i = start; i < end && i < nnz; i++) {
    atomicAdd(&res[row[i]], val[i] * arr[col[i]]);
  }
}
