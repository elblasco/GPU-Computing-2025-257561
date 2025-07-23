#pragma once

#include "utils.cuh"

// __device__ void matrix_edges_cases(NUM_TYPE *d_res,
//                                    const NUM_TYPE *shared_prefix,
//                                    const IDX_TYPE *d_row, const size_t nnz,
//                                    const size_t iteration,
//                                    const size_t offset_from_start) {
//   IDX_TYPE idx = (iteration * blockDim.x) + threadIdx.x;
//   IDX_TYPE row_idx = d_row[offset_from_start + idx];
//   IDX_TYPE row_next_idx = d_row[offset_from_start + idx + 1];
//   if (idx + 1 == blockDim.x || offset_from_start + idx + 1 == nnz) {
//     atomicAdd(&d_res[row_idx], shared_prefix[idx]);
//   } else if (row_idx < row_next_idx) {
//     atomicAdd(&d_res[row_idx], shared_prefix[idx]);
//     atomicAdd(&d_res[row_next_idx], -shared_prefix[idx]);
//   }
// }

__global__ void spmv_with_striding(const IDX_TYPE __restrict__ *row,
                                   const IDX_TYPE __restrict__ *col,
                                   const NUM_TYPE *val, const NUM_TYPE *arr,
                                   NUM_TYPE *res, const size_t nnz,
                                   const size_t portion,
                                   const size_t n_thread) {
  const size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  const size_t start = thread_idx;
  const size_t end = (n_thread * portion) + thread_idx;

  for (size_t i = start; i < end && i < nnz; i += n_thread) {
    atomicAdd(&res[row[i]], val[i] * arr[col[i]]);
  }
}

__global__ void spmv_coo_shfl(const IDX_TYPE *__restrict__ d_rows,
                              const IDX_TYPE *__restrict__ d_cols,
                              const NUM_TYPE *__restrict__ d_vals,
                              const NUM_TYPE *__restrict__ d_array,
                              NUM_TYPE *__restrict__ d_res, const size_t nnz,
                              const size_t cell_per_block,
                              const size_t total_threads) {

  const size_t start = blockIdx.x * cell_per_block + threadIdx.x;
  const size_t end = MIN(nnz, (blockIdx.x + 1) * cell_per_block);

  size_t position_in_line = threadIdx.x & (WARP_SIZE - 1);

  for (IDX_TYPE i = start; i < end; i += blockDim.x) {
    IDX_TYPE row = d_rows[i];
    IDX_TYPE col = d_cols[i];
    NUM_TYPE val = d_vals[i];

    NUM_TYPE product = val * d_array[col];

    // Compare row ids between threads in warp
    for (size_t delta = 1; delta < WARP_SIZE; delta <<= 1) {
      NUM_TYPE prev_row = __shfl_up_sync(FULL_WARP_MASK, row, delta);
      NUM_TYPE prev_product = __shfl_up_sync(FULL_WARP_MASK, product, delta);

      // If the current thread has a lane (warp position) greater top the
      // current offset
      if (position_in_line >= delta && prev_row == row) {
        product += prev_product;
      }
    }

    // The last thread in a row segment writes the result
    size_t row_next_bcast = __shfl_down_sync(FULL_WARP_MASK, row, 1);

    if (row < row_next_bcast || position_in_line + 1 == WARP_SIZE ||
        position_in_line + 1 == nnz) {
      // only one thread writes out to global memory
      atomicAdd(&d_res[row], product);
    }
  }
}

__global__ void spmv_coo_shfl_unroll(
    const IDX_TYPE *__restrict__ d_rows, const IDX_TYPE *__restrict__ d_cols,
    const NUM_TYPE *__restrict__ d_vals, const NUM_TYPE *__restrict__ d_array,
    NUM_TYPE *__restrict__ d_res, const size_t nnz, const size_t cell_per_block,
    const size_t total_threads) {

  const size_t start = blockIdx.x * cell_per_block + threadIdx.x;
  const size_t end = MIN(nnz, (blockIdx.x + 1) * cell_per_block);

  size_t position_in_line = threadIdx.x & (WARP_SIZE - 1);

  for (IDX_TYPE i = start; i < end; i += blockDim.x) {
    IDX_TYPE row = d_rows[i];
    IDX_TYPE col = d_cols[i];
    NUM_TYPE val = d_vals[i];

    NUM_TYPE product = val * d_array[col];

    // Compare row ids between threads in warp
#pragma unroll
    for (size_t delta = 1; delta < WARP_SIZE; delta <<= 1) {
      NUM_TYPE prev_row = __shfl_up_sync(FULL_WARP_MASK, row, delta);
      NUM_TYPE prev_product = __shfl_up_sync(FULL_WARP_MASK, product, delta);

      // If the current thread has a lane (warp position) greater top the
      // current offset
      if (position_in_line >= delta && prev_row == row) {
        product += prev_product;
      }
    }

    // The last thread in a row segment writes the result
    size_t row_next_bcast = __shfl_down_sync(FULL_WARP_MASK, row, 1);

    if (row < row_next_bcast || position_in_line + 1 == WARP_SIZE ||
        position_in_line + 1 == nnz) {
      // only one thread writes out to global memory
      atomicAdd(&d_res[row], product);
    }
  }
}

// Inspired by:
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// __global__ void shared_prefix_sum(const IDX_TYPE *d_row, const IDX_TYPE *d_col,
//                                   const NUM_TYPE *d_val,
//                                   const NUM_TYPE *d_dense_vec, NUM_TYPE *d_res,
//                                   const IDX_TYPE nnz,
//                                   const size_t cell_per_block,
//                                   const size_t tot_threads) {
//   extern __shared__ NUM_TYPE shared_prefix[];

//   const size_t start = blockIdx.x * cell_per_block + threadIdx.x;
//   const size_t end = MIN(nnz, (blockIdx.x + 1) * cell_per_block);

//   for (size_t global_index = start; global_index < nnz;
//        global_index += blockDim.x) {

//     const size_t offset_from_start = blockIdx.x * blockDim.x;
//     const size_t global_thread_id = offset_from_start + threadIdx.x;
//     size_t pout = 0, pin = 1;

//     // Multiplication
//     shared_prefix[threadIdx.x] =
//         d_val[global_index] * d_dense_vec[d_col[global_index]];
//     __syncthreads();

//     // Partial prefix sum
//     for (size_t offset = blockDim.x - 1; offset != 0; offset >>= 1) {
//       // swap double buffer indices
//       pout = 1 - pout;
//       pin = 1 - pout;
//       shared_prefix[pout * blockDim.x + threadIdx.x] +=
//           (threadIdx.x <= offset)
//               ? shared_prefix[pin * blockDim.x + threadIdx.x - offset]
//               : 0;
//       __syncthreads();
//     }

//     if (threadIdx.x + 1 == blockDim.x || global_thread_id + 1 == nnz) {
//       atomicAdd(&d_res[d_row[global_thread_id]],
//                 shared_prefix[pout * blockDim.x + threadIdx.x]);
//     } else if (d_row[global_thread_id] < d_row[global_thread_id + 1]) {
//       atomicAdd(&d_res[d_row[global_thread_id]],
//                 shared_prefix[pout * blockDim.x + threadIdx.x]);
//       atomicAdd(&d_res[d_row[global_thread_id + 1]],
//                 -shared_prefix[pout * blockDim.x + threadIdx.x]);
//     }
//   }
// }

__global__ void prescan(const IDX_TYPE *d_row, const IDX_TYPE *d_col,
                        const NUM_TYPE *d_val, const NUM_TYPE *d_dense_vec,
                        NUM_TYPE *d_res, const IDX_TYPE nnz,
                        const size_t cell_per_block, const size_t tot_threads) {
  extern __shared__ float shared_sum[];
  // allocated on invocation
  const size_t start = blockIdx.x * cell_per_block + threadIdx.x;
  const size_t end = MIN(nnz, (blockIdx.x + 1) * cell_per_block);

  for (size_t global_index = start; global_index < end;
       global_index += blockDim.x) {
    const IDX_TYPE curr_row = d_row[global_index];
    const IDX_TYPE curr_col = d_row[global_index];
    const NUM_TYPE curr_val = d_row[global_index];
    const size_t local_thread_id = threadIdx.x;
    size_t offset = 1;

    shared_sum[local_thread_id] =
        curr_val * d_dense_vec[curr_col]; // load input into shared memory
    // shared_sum[2 * local_thread_id + 1] = g_idata[2 * local_thread_id + 1];

    for (size_t d = blockDim.x - 1; d > 0; d >>= 1) {
      __syncthreads();
      if (local_thread_id < d) {

        // size_t ai = offset * (2 * local_thread_id + 1) - 1;
        const size_t bi =
            offset + local_thread_id; // offset * (2 * local_thread_id + 2) - 1;

        const IDX_TYPE other_row = d_row[blockIdx.x * cell_per_block + bi];

        shared_sum[bi] +=
            (curr_row == other_row) ? shared_sum[local_thread_id] : 0;
      }
      offset <<= 1;
    }

    // The last thread in a row segment writes the result
    const IDX_TYPE row_next_bcast = __shfl_down_sync(FULL_WARP_MASK, curr_row, 1);

    if (curr_row < row_next_bcast || local_thread_id + 1 == blockDim.x ||
        global_index + 1 == nnz) {
      // only one thread writes out to global memory
      atomicAdd(&d_res[curr_row], shared_sum[local_thread_id]);
    }
  }
}
