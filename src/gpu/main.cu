#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#define NROWS 10000
#define NCOLS 10000
#define NNZ 5000
#define MODULO 32
#define BLOCK_SIZE 512

// CUDA error checking
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void sort_coo(size_t *row_idx, size_t *col_idx, double *values,
              size_t nonzeroelem);
double *pre_filled_array(size_t size, double value);
std::tuple<size_t *, size_t *, double *> get_COO();
void array_to_stdout(double *array, size_t rows);
void build_row_ptr_from_coo(size_t *row_idx, size_t *&d_row_offsets_out);
void flops_counter(size_t nnz, float ms);

__global__ void row_with_stride(const size_t *d_row_idx,
                                const size_t *d_col_idx, const double *d_val,
                                const double *d_arr, double *d_res,
                                const size_t num_rows,
                                const size_t *row_offsets);

void test_row_with_stride(const size_t *d_row_indices, const size_t *d_col_indices,
						  const double *d_val, const double *d_arr,
						  const size_t num_rows, const size_t *d_row_offsets,
						  const double *d_test_res, const size_t nnz);

__global__ void row_with_sequential(const size_t *d_row_idx,
                                    const size_t *d_col_idx,
                                    const double *d_val, const double *d_arr,
                                    double *d_res, const size_t num_rows,
                                    const size_t *row_offsets);

void test_row_with_sequential(const size_t *d_row_indices,
                              const size_t *d_col_indices, const double *d_val,
                              const double *d_arr, const size_t num_rows,
                              const size_t *d_row_offsets,
                              const double *d_test_res, const size_t nnz);

__global__ void entire_row(const size_t *row, const size_t *col,
                           const double *val, const double *arr, double *res,
                           const size_t num_rows, const size_t *row_offsets);

void test_entire_row(const size_t *d_row_indices, const size_t *d_col_indices,
                     const double *d_val, const double *d_arr,
                     const size_t num_rows, const size_t *d_row_offsets,
                     const double *d_test_res, const size_t nnz);

__global__ void test_coo_spmv_kernel(const size_t *row, const size_t *col,
                                     const double *val, const double *arr,
                                     double *res, size_t nonzeroelem);

int main(int argc, char **argv) {

  printf("Aboput to create the COO\n");
  auto [d_row_indices, d_col_indices, d_vals] = get_COO();

  double *array1 = pre_filled_array(NCOLS, 1.0f);
  double *cmp_resulting_array = pre_filled_array(NROWS, 0);
  size_t *d_row_offsets;
  build_row_ptr_from_coo(d_row_indices, d_row_offsets);

  //sort_coo(d_row_indices, d_col_indices, d_vals, NNZ);

  size_t gridSizeNNZ = (NNZ + BLOCK_SIZE - 1) / BLOCK_SIZE;

  test_coo_spmv_kernel<<<gridSizeNNZ, BLOCK_SIZE>>>(
      d_row_indices, d_col_indices, d_vals, array1, cmp_resulting_array, NNZ);

  cudaDeviceSynchronize();

  test_entire_row(d_row_indices, d_col_indices, d_vals, array1, NROWS,
                  d_row_offsets, cmp_resulting_array, NNZ);

  // test_row_with_stride(row_indices, col_indices, d_vals, array1, rows,
  // d_row_offsets, cmp_resulting_array, NNZ);

  // test_row_with_sequential(row_indices, col_indices, d_vals, array1, rows,
  // d_row_offsets, cmp_resulting_array, NNZ);

  cudaFree(d_row_offsets);
  cudaFree(d_row_indices);
  cudaFree(d_col_indices);
  cudaFree(d_vals);
  cudaFree(array1);
  cudaFree(cmp_resulting_array);
  return 0;
}

void flops_counter(size_t nnz, float ms) {
  size_t flops = 2 * nnz;
  // printf("We have a total of %lu FP64 operations\n", flops);
  printf("We developed a total of %lf TFLOPs\n", (flops / (ms / 1.e3)) / 1.e12);
}

void test_row_with_sequential(const size_t *d_row_indices,
                              const size_t *d_col_indices, const double *d_val,
                              const double *d_arr, const size_t num_rows,
                              const size_t *d_row_offsets,
                              const double *d_test_res, const size_t nnz) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double *d_resulting_array = pre_filled_array(num_rows, 0);

  size_t gridSizeRowStride = std::ceil((num_rows * MODULO) / BLOCK_SIZE);

  cudaEventRecord(start);

  row_with_sequential<<<gridSizeRowStride, BLOCK_SIZE>>>(
      d_row_indices, d_col_indices, d_val, d_arr, d_resulting_array, num_rows,
      d_row_offsets);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf(
      "Kernel access a single with multiple thread sequentially Time: %f ms\n",
      milliseconds);

  cudaDeviceSynchronize();

  size_t errors = 0;
  for (size_t i = 0; i < num_rows; ++i) {
    errors += d_resulting_array[i] != d_test_res[i];
  }
  printf("Errors: %lu\n", errors);

  flops_counter(nnz, milliseconds);

  array_to_stdout(d_resulting_array, num_rows);

  cudaFree(d_resulting_array);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void test_row_with_stride(const size_t *d_row_indices,
                          const size_t *d_col_indices, const double *d_val,
                          const double *d_arr, const size_t num_rows,
                          const size_t *d_row_offsets, const double *d_test_res,
                          const size_t nnz) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double *d_resulting_array = pre_filled_array(num_rows, 0);

  size_t gridSizeRowStride = std::ceil(MODULO * num_rows / BLOCK_SIZE);

  cudaEventRecord(start);
  row_with_stride<<<gridSizeRowStride, BLOCK_SIZE>>>(
      d_row_indices, d_col_indices, d_val, d_arr, d_resulting_array, num_rows,
      d_row_offsets);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel access a single with multiple thread and stride Time: %f ms\n",
         milliseconds);
  size_t errors = 0;
  for (size_t i = 0; i < num_rows; ++i) {
    errors += d_resulting_array[i] != d_test_res[i];
  }
  printf("Errors: %lu\n", errors);

  flops_counter(nnz, milliseconds);

  array_to_stdout(d_resulting_array, num_rows);

  cudaFree(d_resulting_array);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

__global__ void row_with_stride(const size_t *d_row_idx,
                                const size_t *d_col_idx, const double *d_val,
                                const double *d_arr, double *d_res,
                                const size_t num_rows,
                                const size_t *row_offsets) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t stride = idx % MODULO;
  size_t row_index = idx / MODULO;
  if (row_index >= num_rows)
    return;

  size_t start = row_offsets[row_index] + stride;
  size_t end = row_offsets[row_index + 1];
  double thread_sum = 0;

  for (size_t i = start; i < end; i += MODULO) {
    // res[idx] += val[i] * arr[col[i]];
    thread_sum += d_val[i] * d_arr[d_col_idx[i]];
  }
  d_res[row_index] = thread_sum;
}

__global__ void row_with_sequential(const size_t *d_row_idx,
                                    const size_t *d_col_idx,
                                    const double *d_val, const double *d_arr,
                                    double *d_res, const size_t num_rows,
                                    const size_t *row_offsets) {
  size_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t local_idx = global_idx % MODULO;
  size_t row_index = global_idx / MODULO;
  size_t row_lenght = row_offsets[row_index + 1] - row_offsets[row_index];

  if (row_index >= num_rows || local_idx >= row_lenght)
    return;

  size_t batch_lenght = row_lenght / MODULO + 1;
  size_t start = row_offsets[row_index] + batch_lenght * local_idx;
  size_t end = start + batch_lenght;
  double thread_sum = 0;

  for (size_t i = start; i < end || i < row_lenght; i++) {
    thread_sum += d_val[i] * d_arr[d_col_idx[i]];
  }
  d_res[row_index] += thread_sum;
}

__global__ void entire_row(const size_t *row, const size_t *col,
                           const double *val, const double *arr, double *res,
                           const size_t num_rows, const size_t *row_offsets) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_rows)
    return;

  size_t start = row_offsets[idx];
  size_t end = row_offsets[idx + 1];

  double sum = 0;

  for (size_t i = start; i < end; ++i) {
    sum += val[i] * arr[col[i]];
  }

  res[idx] = sum;
}

__global__ void test_coo_spmv_kernel(const size_t *row, const size_t *col,
                                     const double *val, const double *arr,
                                     double *res, size_t nonzeroelem) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nonzeroelem) {
    size_t r = row[i];
    size_t c = col[i];
    double v = val[i];
    atomicAdd(&res[r], v * arr[c]);
  }
}

void build_row_ptr_from_coo(size_t *d_row_idx, // COO row indices [nnz]
                            size_t *&d_row_offsets_out) {

  std::vector<size_t> row_offsets_host(NROWS + 1, 0);
  size_t *h_row_idx = new size_t(NROWS);
  cudaMemcpy(h_row_idx, d_row_idx, NROWS * sizeof(size_t), cudaMemcpyDeviceToHost);
  // Step 1: Count number of nonzeros per row
  for (size_t i = 0; i < NNZ; ++i)
    row_offsets_host[h_row_idx[i] + 1]++;

  // Step 2: Cumulative sum to get row_ptr
  for (size_t i = 0; i <= NROWS; ++i)
    row_offsets_host[i + 1] += row_offsets_host[i];

  delete[] h_row_idx;
  cudaMalloc(&d_row_offsets_out, (NROWS + 1) * sizeof(size_t));
  cudaMemcpy(d_row_offsets_out, row_offsets_host.data(),
             (NROWS + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
}

void sort_coo(size_t *row_idx, size_t *col_idx, double *values,
              size_t nonzeroelem) {
  size_t *indices = new size_t[nonzeroelem];
  for (size_t i = 0; i < nonzeroelem; ++i)
    indices[i] = i;

  std::sort(indices, indices + nonzeroelem, [&](int a, int b) {
    if (row_idx[a] != row_idx[b])
      return row_idx[a] < row_idx[b];
    return col_idx[a] < col_idx[b];
  });

  size_t *sorted_row = new size_t[nonzeroelem];
  size_t *sorted_col = new size_t[nonzeroelem];
  double *sorted_val = new double[nonzeroelem];

  for (int i = 0; i < nonzeroelem; ++i) {
    int idx = indices[i];
    sorted_row[i] = row_idx[idx];
    sorted_col[i] = col_idx[idx];
    sorted_val[i] = values[idx];
  }

  for (int i = 0; i < nonzeroelem; ++i) {
    row_idx[i] = sorted_row[i];
    col_idx[i] = sorted_col[i];
    values[i] = sorted_val[i];
  }

  delete[] indices;
  delete[] sorted_row;
  delete[] sorted_col;
  delete[] sorted_val;
}

std::tuple<size_t *, size_t *, double *> get_COO() {
  srand((unsigned int)time(NULL));

  size_t count = 0;
  size_t *h_row_indices = new size_t[NNZ];
  size_t *h_col_indices = new size_t[NNZ];
  double *h_vals = new double[NNZ];
  size_t *d_row_indices, *d_col_indices;
  double *d_vals;

    // Track used positions (row-major indexing)
  int used[NROWS * NCOLS] = {0};
  
  for (int r = 0; r < NROWS && count < NNZ; ++r) {
	for (int c = 0; c < NCOLS && count < NNZ; ++c) {
	  int index = r * NCOLS + c;
	  if (!used[index] && (rand() % 2)) { // Randomly decide to place a non-zero
		used[index] = 1;
		h_row_indices[count] = r;
		h_col_indices[count] = c;
		h_vals[count] = (double)(rand() % 100 + 1);
		count++;
	  }
	}
  }

  for (int r = 0; r < NROWS && count < NNZ; ++r) {
    for (int c = 0; c < NCOLS && count < NNZ; ++c) {
      int index = r * NCOLS + c;
      if (!used[index]) {
        used[index] = 1;
        h_row_indices[count] = r;
        h_col_indices[count] = c;
        h_vals[count] = (double)(rand() % 100 + 1);
        count++;
      }
    }
  }

  cudaCheckError(cudaMalloc(&d_row_indices, NNZ * sizeof(size_t)));
  cudaCheckError(cudaMalloc(&d_col_indices, NNZ * sizeof(size_t)));
  cudaCheckError(cudaMalloc(&d_vals, NNZ * sizeof(double)));

  cudaCheckError(cudaMemcpy(d_row_indices, h_row_indices, NNZ * sizeof(size_t),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_col_indices, h_col_indices, NNZ * sizeof(size_t),
                            cudaMemcpyHostToDevice));
  cudaCheckError(
      cudaMemcpy(d_vals, h_vals, NNZ * sizeof(double), cudaMemcpyHostToDevice));

  delete[] h_vals;
  delete[] h_col_indices;
  delete[] h_row_indices;

  return {d_row_indices, d_col_indices, d_vals};
}

void array_to_stdout(double *array, size_t rows) {
  std::cout << "This is the encoding of the result" << std::endl;
  std::cout << "The first line is the number of rows" << std::endl;
  std::cout << "Each line is a cell of the array" << std::endl;

  std::cout << rows << std::endl;

  for (size_t row = 0; row < rows; ++row) {
    std::cout << std::setprecision(15) << array[row] << std::endl;
  }
}

double *pre_filled_array(size_t size, double value) {
  double* h_arr = new double[size];
  for (size_t i = 0; i < size; ++i) {
    h_arr[i] = value;
  }
  double * d_arr;
  cudaMalloc(&d_arr, size * sizeof(double));
  cudaMemcpy(d_arr, h_arr, size * sizeof(double), cudaMemcpyHostToDevice);
  delete[] h_arr;
  printf("Prefilled array done\n");
  return d_arr;
}

void print_double_bits(double value) {
    union {
        double d;
        uint64_t u;
    } converter;

    converter.d = value;

    for (int i = 63; i >= 0; --i) {
	  
        printf("%d", (converter.u >> i) & 1);
        if (i % 4 == 0) printf(" "); // optional: space every 4 bits
	  }
    printf("(%.15f)\n", value);
}

void test_entire_row(const size_t *d_row_indices, const size_t *d_col_indices,
                     const double *d_val, const double *d_arr,
                     const size_t num_rows, const size_t *d_row_offsets,
                     const double *d_test_res, const size_t nnz) {

  //for(size_t i = 0; i < num_rows; ++i)
  //printf("%lu ", d_row_offsets[i]);

  printf("I am launching the one thread one row kernel\n");
  printf("It should uses %lu threads\n", num_rows);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double *d_resulting_array = pre_filled_array(num_rows, 0);
  double *h_resulting_array = NULL, *h_test_res = NULL;

  size_t gridSizeRowNoStride = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  printf("The number of thread invoked launching the kernle is %lu\n", BLOCK_SIZE * gridSizeRowNoStride);
  cudaEventRecord(start);
  entire_row<<<gridSizeRowNoStride, BLOCK_SIZE>>>(
      d_row_indices, d_col_indices, d_val, d_arr, d_resulting_array, num_rows,
      d_row_offsets);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel which takes a row for each thread Time: %f ms\n", milliseconds);
  
  cudaDeviceSynchronize();

  cudaMemcpy(h_resulting_array, d_resulting_array, num_rows * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_test_res, d_test_res, num_rows * sizeof(double), cudaMemcpyDeviceToHost);
  
  size_t errors = 0;
  for (size_t i = 0; i < num_rows; ++i) {
    errors += h_resulting_array[i] != h_test_res[i];
	if(h_resulting_array[i] != h_test_res[i] && std::abs(h_resulting_array[i] - h_test_res[i]) > 0.00000001){
	  std::cout << "Error on row " << i << std::endl;
	print_double_bits(h_resulting_array[i]);
	print_double_bits(h_test_res[i]);
	}
  }
  printf("Errors: %lu\n", errors);

  flops_counter(nnz, milliseconds);
  
  delete[] h_resulting_array;
  delete[] h_test_res;
  cudaFree(d_resulting_array);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
