#pragma once

#include "utils.cuh"

__global__ void spmv_with_striding(const IDX_TYPE __restrict__ *row,
                                   const IDX_TYPE __restrict__ *col,
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

__global__ void spmv_coo_kernel_shared(const IDX_TYPE __restrict__ *d_row,
                                       const IDX_TYPE __restrict__ *d_col,
                                       const NUM_TYPE *values,
                                       const NUM_TYPE *x, NUM_TYPE *y,
                                       const NUM_TYPE nnz,
                                       const NUM_TYPE num_rows) {
  __shared__ NUM_TYPE shared_pref_sum[MAX_BLOCK_SIZE];

  size_t id_thread = threadIdx.x + blockIdx.x * blockDim.x;
  size_t n_thread = blockDim.x * gridDim.x;

  // Initialize shared memory
  if (threadIdx.x < MAX_BLOCK_SIZE) {
    shared_pref_sum[threadIdx.x] = 0.0f;
  }

  __syncthreads();

  for (size_t idx = id_thread; idx < nnz; idx += n_thread) {
    IDX_TYPE row = d_row[idx];
    IDX_TYPE col = d_col[idx];
    NUM_TYPE val = values[idx];

    NUM_TYPE product = val * x[col];

    // Each thread writes its product in shared memory for reduction
    atomicAdd(&shared_pref_sum[idx % MAX_BLOCK_SIZE], product);
  }

  //__syncthreads();

  // Cooperative intra-warp reduction:

  // Use the first thread in the block for each unique row to accumulate
  // partial sums This reduces the number of atomicAdds
  // float sum = sdata[threadIdx.x];

  // atomicAdd on global y
  // atomicAdd(&y[row], sum);
}

__global__ void spmv_coo_shfl_unroll(const IDX_TYPE *__restrict__ d_rows,
                                     const IDX_TYPE *__restrict__ d_cols,
                                     const NUM_TYPE *__restrict__ d_vals,
                                     const NUM_TYPE *__restrict__ d_array,
                                     NUM_TYPE *__restrict__ d_res,
                                     IDX_TYPE nnz) {
  size_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  size_t lane = threadIdx.x % WARP_SIZE;

  size_t total_threads = gridDim.x * blockDim.x;

  for (size_t i = thread_id; i < nnz; i += total_threads) {
    IDX_TYPE row = d_rows[i];
    IDX_TYPE col = d_cols[i];
    NUM_TYPE val = d_vals[i];

    NUM_TYPE product = val * d_array[col];

// Compare row ids between threads in warp
#pragma unroll
    // Compare row ids between threads in warp
    for (size_t delta = 1; delta < WARP_SIZE; delta <<= 2) {
      NUM_TYPE prev_row = __shfl_up_sync(FULL_WARP_MASK, row, delta);
      NUM_TYPE prev_product = __shfl_up_sync(FULL_WARP_MASK, product, delta);

      // If the current thread has a lane (warp position) greater top the
      // current offset
      if (lane >= delta && prev_row == row) {
        product += prev_product;
      }
    }

    // The last thread in a row segment writes the result
    size_t row_next_bcast = __shfl_down_sync(FULL_WARP_MASK, row, 1);

    if (row != row_next_bcast) {
      // only one thread writes out to global memory
      atomicAdd(&d_res[row], product);
    }
  }
}

__global__ void spmv_coo_shfl(const IDX_TYPE *__restrict__ d_rows,
                              const IDX_TYPE *__restrict__ d_cols,
                              const NUM_TYPE *__restrict__ d_vals,
                              const NUM_TYPE *__restrict__ d_array,
                              NUM_TYPE *__restrict__ d_res, IDX_TYPE nnz) {
  size_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  size_t lane = threadIdx.x % WARP_SIZE;

  size_t total_threads = gridDim.x * blockDim.x;

  for (size_t i = thread_id; i < nnz; i += total_threads) {
    IDX_TYPE row = d_rows[i];
    IDX_TYPE col = d_cols[i];
    NUM_TYPE val = d_vals[i];

    NUM_TYPE product = val * d_array[col];

    // Compare row ids between threads in warp
    for (size_t delta = 1; delta < WARP_SIZE; delta <<= 2) {
      NUM_TYPE prev_row = __shfl_up_sync(FULL_WARP_MASK, row, delta);
      NUM_TYPE prev_product = __shfl_up_sync(FULL_WARP_MASK, product, delta);

      // If the current thread has a lane (warp position) greater top the
      // current offset
      if (lane >= delta && prev_row == row) {
        product += prev_product;
      }
    }

    // The last thread in a row segment writes the result
    size_t row_next_bcast = __shfl_down_sync(FULL_WARP_MASK, row, 1);

    if (row != row_next_bcast) {
      // only one thread writes out to global memory
      atomicAdd(&d_res[row], product);
    }
  }
}
