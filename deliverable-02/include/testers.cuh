#pragma once

#include "utils.cuh"

void test_spmv_gpu(gpu_kernel kernel, const COO_local<uint32_t, NUM_TYPE> *sparse_matrix) {
  float times[NUM_TEST];
  float flops[NUM_TEST];
  float bandwidth[NUM_TEST];
  size_t nnz = sparse_matrix -> nnz;

  CUDA_MANGED_MALLOC(d_vals, NUM_TYPE, nnz);
  CUDA_MANGED_MALLOC(d_rows, IDX_TYPE, nnz);
  CUDA_MANGED_MALLOC(d_cols, IDX_TYPE, nnz);

  for (size_t i = 0; i < nnz; ++i){
	d_vals[i] = sparse_matrix -> val[i];
	d_rows[i] = sparse_matrix -> row[i];
	d_cols[i] = sparse_matrix -> col[i];
  }

  CUDA_TIMER_DEF(gpu_time);

  size_t possible_grid_size = MAX_GRID_SIZE;
  size_t possible_block_size = MAX_BLOCK_SIZE;

  size_t grid_size = MIN(possible_grid_size, (size_t)std::ceil(nnz / (float)MAX_BLOCK_SIZE));
  size_t block_size = MIN(possible_block_size, nnz);

  size_t portion = CEILING(nnz, (block_size * grid_size));
  printf("The kernl will be executed on %lu threads, eaach of them should cover at most %lu elements\n", (grid_size * block_size), portion);

  for (size_t run = 0; run < NUM_TEST + WARM_UP_RUN; run++) {
    CUDA_MANGED_MALLOC(d_res_array, NUM_TYPE, sparse_matrix -> nrows);
	CUDA_MANGED_MALLOC(d_dense_array, NUM_TYPE, sparse_matrix -> nrows);

	printf("Iteration %lu\n", run);

	for (size_t j = 0; j < sparse_matrix -> nrows; ++j){
	  d_res_array[j] = 0.0;
	  d_dense_array[j] = 1.0;
	}

	CUDA_TIMER_START(gpu_time);

	kernel<<<grid_size, block_size>>>(d_rows, d_cols, d_vals, d_dense_array, d_res_array, nnz, portion);
    cudaDeviceSynchronize();

	CUDA_TIMER_STOP(gpu_time);

	float milliseconds = 0;
	CUDA_TIMER_ELAPSED(gpu_time);

	if (WARM_UP_RUN <= run){
	  flops[run - WARM_UP_RUN] = flops_counter(nnz, milliseconds);
	  times[run - WARM_UP_RUN] = milliseconds;
	  bandwidth[run - WARM_UP_RUN] = (nnz * sizeof(NUM_TYPE) * MEMEORY_RW / milliseconds) / 1e12;
	}

	CUDA_FREE(d_dense_array);
	CUDA_FREE(d_res_array);
  }

  double times_mu = geometric_mean(times, NUM_TEST);
  double times_sigma = sigma_fn(times, times_mu, NUM_TEST);

  printf("This kernel executed with an average of %lf ms with std.dev. of %lf ms\n", times_mu, times_sigma);

  double flops_mu = geometric_mean(flops, NUM_TEST);
  double flops_sigma = sigma_fn(flops, flops_mu, NUM_TEST);

  printf("This kernel produced an average of %lf GFLOP/s with std.dev. of %lf GFLOP/s\n", flops_mu, flops_sigma);

  double bandwidth_mu = geometric_mean(bandwidth, NUM_TEST);
  double bandwidth_sigma = sigma_fn(bandwidth, bandwidth_mu, NUM_TEST);

  printf("This kernel produced an avergare bandwidth of %lf GB/s with std.dev. of %lf GB/s the theoretical maximun is 933 GB/s\n", bandwidth_mu, bandwidth_sigma);

  CUDA_TIMER_DESTROY(gpu_time);
  CUDA_FREE(d_vals);
  CUDA_FREE(d_rows);
  CUDA_FREE(d_cols);
}
