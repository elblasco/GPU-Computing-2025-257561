#pragma once

#include "utils.cuh"

__device__ void matrix_edges_cases(NUM_TYPE *d_res,
                                   const NUM_TYPE *shared_prefix,
                                   const IDX_TYPE *d_row, const size_t nnz,
                                   const size_t iteration,
                                   const size_t offset_from_start) {
  IDX_TYPE idx = (iteration * blockDim.x) + threadIdx.x;
  IDX_TYPE row_idx = d_row[offset_from_start + idx];
  IDX_TYPE row_next_idx = d_row[offset_from_start + idx + 1];
  if (idx + 1 == blockDim.x || offset_from_start + idx + 1 == nnz) {
    atomicAdd(&d_res[row_idx], shared_prefix[idx]);
  } else if (row_idx < row_next_idx) {
    atomicAdd(&d_res[row_idx], shared_prefix[idx]);
    atomicAdd(&d_res[row_next_idx], -shared_prefix[idx]);
  }
}

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

#define ELEM_PER_THREAD 2
__global__ void shared_prefix_sum(const IDX_TYPE *d_row, const IDX_TYPE *d_col,
                                  const NUM_TYPE *d_val,
                                  const NUM_TYPE *d_dense_vec,
                                  NUM_TYPE *d_result, const IDX_TYPE nnz) {
  extern __shared__ NUM_TYPE shared_prefix[];
  size_t read_per_block = blockDim.x * ELEM_PER_THREAD;

  // Each block does `read_per_block` consecutive elements
  size_t offset_from_start = blockIdx.x * read_per_block;

  // Each thread multiplies `ELEM_PER_THREAD` into shared memory
  for (size_t i = 0; i < ELEM_PER_THREAD; i++) {
    size_t idx = (i * blockDim.x) + threadIdx.x;
    if (offset_from_start + idx < nnz) {
      // `d_col[offset_from_start + idx]` is the current column
      shared_prefix[idx] = d_val[offset_from_start + idx] *
                           d_dense_vec[d_col[offset_from_start + idx]];
    } else {
      shared_prefix[idx] = 0;
    }
  }

  __syncthreads();

  // For every thread in the block (except the first)
  // iterate over its sum and
  for (IDX_TYPE s = 1; s < (read_per_block / ELEM_PER_THREAD); s <<= 1) {
    for (size_t i = 0; i < ELEM_PER_THREAD; i++) {
      size_t idx = (i * blockDim.x) + threadIdx.x;
      if (idx + s < read_per_block) {
        // printf("Thread %d has a shared sum of %f at index %lu (the original
        // "
        //        "matrix was [%lu][%lu] with value %f), now it will update "
        //        "shared with index %u which has value of %f to value %f\n",
        //        threadIdx.x, shared_prefix[idx], idx,
        //        d_row[offset_from_start + idx], d_col[offset_from_start +
        //        idx], d_val[offset_from_start + idx], s + idx,
        //        shared_prefix[idx + s], shared_prefix[idx + s] +
        //        shared_prefix[idx]);
        shared_prefix[idx + s] += shared_prefix[idx];
      }
    }
    __syncthreads();
  }

  // Memory write
  for (size_t i = 0; i < ELEM_PER_THREAD; i++) {
    matrix_edges_cases(d_result, shared_prefix, d_row, nnz, i,
                       offset_from_start);
  }
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
