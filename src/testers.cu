#include <chrono>
#include <ratio>
#include <cmath>

#include "include/cpu.h"
#include "include/utils.h"
#include "include/macros.h"

NUM_TYPE* test_spmv_cpu(const IDX_TYPE *row_indices, const IDX_TYPE *col_indices,
                   const NUM_TYPE *val, const NUM_TYPE *arr,
                   const IDX_TYPE num_rows, const size_t nnz) {
  double times[NUM_TEST];
  double flops[NUM_TEST];
  double bandwidth[NUM_TEST];
  NUM_TYPE *resulting_array;
  
  for (size_t i = 0; i < NUM_TEST; ++i) {
	resulting_array = pre_filled_array_cpu(num_rows, 0);
	
	auto start = std::chrono::high_resolution_clock::now(); 
    spmv_cpu(row_indices, col_indices, val, arr, nnz, resulting_array);
    auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> duration_ms = end - start;
	
    double elapsed_ms = duration_ms.count();
	flops[i] = flops_counter(nnz, elapsed_ms);
	times[i] = elapsed_ms;
	bandwidth[i] = (nnz * sizeof(NUM_TYPE) * MEMEORY_RW / elapsed_ms) / 1e12;
	
	if(i < NUM_TEST - 1)
	  delete[] resulting_array;
  }
  
  double times_mu = mu_fn(times, NUM_TEST);
  double times_sigma = sigma_fn(times, times_mu, NUM_TEST);
  
  printf("This kernel executed with an average of %lf ms with std.dev. of %lf ms\n", times_mu, times_sigma);
  
  double flops_mu = mu_fn(flops, NUM_TEST);
  double flops_sigma = sigma_fn(flops, flops_mu, NUM_TEST);
  
  printf("This kernel produced an average of %lf GFLOPS with std.dev. of %lf GFLOPS\n", flops_mu, flops_sigma);

  double bandwidth_mu = mu_fn(bandwidth, NUM_TEST);
  double bandwidth_sigma = sigma_fn(bandwidth, bandwidth_mu, NUM_TEST);
  
  printf("This kernel produced an avergare bandwidth of %lf GB/ms with std.dev. of %lf GB/ms\n", bandwidth_mu, bandwidth_sigma);
  return resulting_array;
}

NUM_TYPE* test_spmv_gpu(gpu_kernel kernel, const IDX_TYPE *row_indices,
               const IDX_TYPE *col_indices, const NUM_TYPE *val,
               const NUM_TYPE *arr, const IDX_TYPE num_rows, const size_t nnz) {
  double times[NUM_TEST];
  double flops[NUM_TEST];
  double bandwidth[NUM_TEST];

  cudaEvent_t start, stop;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));

  NUM_TYPE *resulting_array;

  size_t possible_grid_size = MAX_GRID_SIZE;
  size_t possible_block_size = MAX_BLOCK_SIZE;

  size_t grid_size = std::min(possible_grid_size, (size_t)std::ceil(nnz / (float)MAX_BLOCK_SIZE));
  size_t block_size = std::min(possible_block_size, nnz);

  size_t portion = std::ceil(nnz / (block_size * grid_size));
  printf("The kernl will be executed on %lu threads, eaach of them should cover at most %lu elements\n", (grid_size * block_size), portion);

  for (size_t i = 0; i < NUM_TEST; ++i) {
	resulting_array = pre_filled_array_gpu(num_rows, 0);
    cudaCheckError(cudaEventRecord(start));
    kernel<<<grid_size, block_size>>>(row_indices, col_indices, val, arr, resulting_array, nnz, portion);
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));
    cudaCheckError(cudaDeviceSynchronize());

    float milliseconds = 0;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));
	
    flops[i] = flops_counter(nnz, milliseconds);
    times[i] = milliseconds;
	bandwidth[i] = (nnz * sizeof(NUM_TYPE) * MEMEORY_RW / milliseconds) / 1e12;
	
	if(i < NUM_TEST - 1)
	  cudaCheckError(cudaFree(resulting_array));
  }
  
  double times_mu = mu_fn(times, NUM_TEST);
  double times_sigma = sigma_fn(times, times_mu, NUM_TEST);
  
  printf("This kernel executed with an average of %lf ms with std.dev. of %lf ms\n", times_mu, times_sigma);
  
  double flops_mu = mu_fn(flops, NUM_TEST);
  double flops_sigma = sigma_fn(flops, flops_mu, NUM_TEST);
  
  printf("This kernel produced an average of %lf GFLOP/s with std.dev. of %lf GFLOP/s\n", flops_mu, flops_sigma);

  double bandwidth_mu = mu_fn(bandwidth, NUM_TEST);
  double bandwidth_sigma = sigma_fn(bandwidth, bandwidth_mu, NUM_TEST);
  
  printf("This kernel produced an avergare bandwidth of %lf GB/s with std.dev. of %lf GB/s the theoretical maximun is 933 GB/s\n", bandwidth_mu, bandwidth_sigma);
  
  cudaCheckError(cudaEventDestroy(start));
  cudaCheckError(cudaEventDestroy(stop));

  return resulting_array;
}
