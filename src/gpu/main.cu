#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

#define MODULO 32
#define FULL_MASK 0xffffffff
#define BLOCK_SIZE 512

void sort_coo(size_t *row_idx, size_t *col_idx, double *values,
              size_t nonzeroelem);
double *pre_filled_array(size_t size, double value);
std::tuple<size_t *, size_t *, double *, size_t, size_t, size_t>
get_COO(const char *file_name);
void array_to_stdout(double *array, size_t rows);
void build_row_ptr_from_coo(size_t *row_idx,
                            size_t **d_row_offsets_out, size_t nnz,
                            size_t num_rows);
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

  if (argc != 2) {
    std::cout << "Usage output.exec <matrix-file>" << std::endl;
    return 1;
  }

  auto [row_indices, col_indices, d_vals, rows, cols, non_zero_count] =
      get_COO(argv[1]);
  double *array1 = pre_filled_array(cols, 1.0f);
  double *cmp_resulting_array = pre_filled_array(rows, 0);
  size_t *d_row_offsets;
  build_row_ptr_from_coo(row_indices, &d_row_offsets, non_zero_count, rows);

  sort_coo(row_indices, col_indices, d_vals, non_zero_count);

  size_t gridSizeNNZ = (non_zero_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  test_coo_spmv_kernel<<<gridSizeNNZ, BLOCK_SIZE>>>(
      row_indices, col_indices, d_vals, array1, cmp_resulting_array,
      non_zero_count);
  
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("The test kernel developed ");
  flops_counter(non_zero_count, milliseconds);

  test_entire_row(row_indices, col_indices, d_vals, array1, rows, d_row_offsets, cmp_resulting_array, non_zero_count);

  //test_row_with_stride(row_indices, col_indices, d_vals, array1, rows, d_row_offsets, cmp_resulting_array, non_zero_count);

  test_row_with_sequential(row_indices, col_indices, d_vals, array1, rows, d_row_offsets, cmp_resulting_array, non_zero_count);

  cudaFree(d_row_offsets);
  cudaFree(row_indices);
  cudaFree(col_indices);
  cudaFree(d_vals);
  cudaFree(array1);
  cudaFree(cmp_resulting_array);
  return 0;
}

void flops_counter(size_t nnz, float ms) {
  size_t flops = 2 * nnz;
  //printf("We have a total of %lu FP64 operations\n", flops);
  printf("We developed a total of %lf TFLOPs\n",
         (flops / (ms / 1.e3)) / 1.e12);
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
  printf("Kernel access a single with multiple thread sequentially Time: %f ms\n", milliseconds);
  
  cudaDeviceSynchronize();
  
  size_t errors = 0;
  for (size_t i = 0; i < num_rows; ++i) {
    errors += d_resulting_array[i] != d_test_res[i];
  }
  printf("Errors: %lu\n", errors);

  flops_counter(nnz, milliseconds);
  
  cudaFree(d_resulting_array);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void test_row_with_stride(const size_t *d_row_indices, const size_t *d_col_indices,
                     const double *d_val, const double *d_arr,
                     const size_t num_rows, const size_t *d_row_offsets,
						  const double *d_test_res, const size_t nnz) {

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
  printf("Kernel access a single with multiple thread and stride Time: %f ms\n", milliseconds);
  size_t errors = 0;
  for (size_t i = 0; i < num_rows; ++i) {
    errors += d_resulting_array[i] != d_test_res[i];
  }
  printf("Errors: %lu\n", errors);

  flops_counter(nnz, milliseconds);
  
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

  for (size_t i = start; i < end ||  i < row_lenght; i++) {
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
  for (size_t i = start; i < end; ++i) {
    res[idx] += val[i] * arr[col[i]];
  }
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

void build_row_ptr_from_coo(size_t *row_idx, // COO row indices [nnz]
                            size_t **d_row_offsets_out, size_t nnz,
                            size_t num_rows) {

  std::vector<size_t> row_offsets_host(num_rows + 1, 0);

  // Step 1: Count number of nonzeros per row
  for (size_t i = 0; i < nnz; ++i)
    row_offsets_host[row_idx[i]]++;

  // Step 2: Cumulative sum to get row_ptr
  for (size_t i = 0; i < num_rows; ++i)
    row_offsets_host[i + 1] += row_offsets_host[i];

  cudaMalloc(d_row_offsets_out, (num_rows + 1) * sizeof(size_t));
  cudaMemcpy(d_row_offsets_out, row_offsets_host.data(),
             (num_rows + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
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

std::tuple<size_t *, size_t *, double *, size_t, size_t, size_t>
get_COO(const char *file_name) {
  std::ifstream MyFile(file_name);
  std::string line;
  // Skip header/comments
  do {
    std::getline(MyFile, line);
  } while (line[0] == '%');

  size_t rows, cols, nonzero_vals;
  size_t *row_indices;
  size_t *col_indices;
  double *vals;
  std::istringstream sizes(line);
  sizes >> rows >> cols >> nonzero_vals;

  cudaMallocManaged(&row_indices, nonzero_vals * sizeof(size_t));
  cudaMallocManaged(&col_indices, nonzero_vals * sizeof(size_t));
  cudaMallocManaged(&vals, nonzero_vals * sizeof(double));

  for (size_t i = 0; i < nonzero_vals; ++i) {
    size_t row, col;
    double val;
    MyFile >> row;
    MyFile >> col;
    MyFile >> val;
    row_indices[i] = row - 1;
    col_indices[i] = col - 1;
    vals[i] = val;
  }

  MyFile.close();
  return {row_indices, col_indices, vals, rows, cols, nonzero_vals};
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
  double *res;
  cudaMallocManaged(&res, size * sizeof(double));
  for (size_t i = 0; i < size; ++i) {
    res[i] = value;
  }
  return res;
}

void test_entire_row(const size_t *d_row_indices, const size_t *d_col_indices,
                     const double *d_val, const double *d_arr,
                     const size_t num_rows, const size_t *d_row_offsets,
                     const double *d_test_res, const size_t nnz) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double *d_resulting_array = pre_filled_array(num_rows, 0);

  size_t gridSizeRowNoStride = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

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
  
  size_t errors = 0;
  for (size_t i = 0; i < num_rows; ++i) {
    errors += d_resulting_array[i] != d_test_res[i];
  }
  printf("Errors: %lu\n", errors);

  flops_counter(nnz, milliseconds);
  
  cudaFree(d_resulting_array);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
