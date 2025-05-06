#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

void sort_coo(size_t *row_idx, size_t *col_idx, double *values,
              size_t nonzeroelem);
double *pre_filled_array(size_t size, double value);
std::tuple<size_t *, size_t *, double *, size_t, size_t, size_t>
get_COO(const char *file_name);
void array_to_stdout(double *array, size_t rows);
void build_row_ptr_from_coo(
							size_t *row_idx, // COO row indices [nnz]
							size_t** d_row_offsets_out,
							size_t nnz, size_t num_rows);
__global__ void coo_spmv_kernel(const size_t *row, const size_t *col,
                                const double *val, const double *arr,
                                double *res, const size_t num_rows,
								const size_t *row_offsets);
__global__ void test_coo_spmv_kernel(const size_t *row, const size_t *col,
                                     const double *val, const double *arr,
                                     double *res, size_t nonzeroelem);

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "Usage output.exec <matrix-file>" << std::endl;
    return 1;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  auto [row_indices, col_indices, vals, rows, cols, non_zero_count] =
      get_COO(argv[1]);
  double *array1 = pre_filled_array(cols, 1.0f);
  double *resulting_array = pre_filled_array(rows, 0);
  double *cmp_resulting_array = pre_filled_array(rows, 0);
  size_t* d_row_offsets;
  build_row_ptr_from_coo(row_indices, &d_row_offsets, non_zero_count, rows);

  sort_coo(row_indices, col_indices, vals, non_zero_count);
  std::cout << "COO sorted" << std::endl;
  size_t blockSize = 256;
  size_t gridSizeRow = (rows + blockSize - 1) / blockSize;
  size_t gridSizeNNZ = (non_zero_count + blockSize - 1) / blockSize;

  cudaEventRecord(start);
  // coo_spmv_kernel<<<gridSizeRow, blockSize>>>(row_indices, col_indices, vals,
  //                                             array1, resulting_array, rows,
  //                                             d_row_offsets);
  
  test_coo_spmv_kernel<<<gridSizeNNZ, blockSize>>>(row_indices, col_indices, vals,
                                                array1, cmp_resulting_array,
                                                non_zero_count);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaDeviceSynchronize();

  // size_t errors = 0;
  // for (size_t i = 0; i < rows; ++i) {
  //   errors += resulting_array[i] != cmp_resulting_array[i];
  // }

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel Time: %f ms\n", milliseconds);
  //printf("Errors: %lu\n", errors);
  size_t flops = 2 * non_zero_count;
  printf("We have a total of %lu FP64 operations\n", flops);
  printf("We developed a total of %lf TFLOPs\n",
         (flops / (milliseconds / 1.e3))/ 1.e12);
  // array_to_stdout(resulting_array, rows);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_row_offsets);
  cudaFree(row_indices);
  cudaFree(col_indices);
  cudaFree(vals);
  cudaFree(array1);
  cudaFree(resulting_array);
  cudaFree(cmp_resulting_array);
  return 0;
}

__global__ void coo_spmv_kernel(const size_t *row, const size_t *col,
                                const double *val, const double *arr,
                                double *res, const size_t num_rows,
								const size_t *row_offsets) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_rows) return;

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
  // if (i < 5) printf("Thread %lu, row=%lu, col=%lu, val=%.2lf,
  // arr[col]=%.2lf\n", i, row[i], col[i], val[i], arr[col[i]]);
  if (i < nonzeroelem) {
    size_t r = row[i];
    size_t c = col[i];
    double v = val[i];
    atomicAdd(&res[r], v * arr[c]);
  }
}

void build_row_ptr_from_coo(
    size_t *row_idx, // COO row indices [nnz]
	size_t** d_row_offsets_out,
    size_t nnz, size_t num_rows) {
  
  std::vector<size_t> row_offsets_host(num_rows + 1, 0);
  // Step 1: Initialize row_ptr to 0
  for (size_t i = 0; i <= num_rows; ++i)
    row_offsets_host[i] = 0;

  // Step 2: Count number of nonzeros per row
  for (size_t i = 0; i < nnz; ++i)
    row_offsets_host[row_idx[i] + 1]++;

  // Step 3: Cumulative sum to get row_ptr
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

double *pre_filled_array(size_t size, double value) {
  double* res;
  cudaMallocManaged(&res, size * sizeof(double));
  for(size_t i = 0; i < size; ++i){
	res[i] = value;
  }
  return res;
}

std::tuple<size_t*, size_t*, double*, size_t, size_t, size_t> get_COO(const char* file_name) {
  std::ifstream MyFile(file_name);
  std::string line;
  // Skip header/comments
  do {
	std::getline(MyFile, line);
  } while (line[0] == '%');

  size_t rows, cols, nonzero_vals;
  size_t* row_indices;
  size_t* col_indices;
  double* vals;
  std::istringstream sizes(line);
  sizes >> rows >> cols >> nonzero_vals;

  cudaMallocManaged(&row_indices, nonzero_vals * sizeof(size_t));
  cudaMallocManaged(&col_indices, nonzero_vals * sizeof(size_t));
  cudaMallocManaged(&vals, nonzero_vals * sizeof(double));
  
  for(size_t i = 0; i < nonzero_vals; ++i){
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

void array_to_stdout(double* array, size_t rows) {
  std::cout << "This is the encoding of the result" << std::endl;
  std::cout << "The first line is the number of rows" << std::endl;
  std::cout << "Each line is a cell of the array" << std::endl;
  
  std::cout << rows << std::endl;
  
  for (size_t row = 0; row < rows; ++row) {
    std::cout << std::setprecision (15) << array[row] << std::endl;
  }
}
