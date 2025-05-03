#include <cstddef>
#include <iostream>
#include <fstream>
#include <tuple>
#include <sstream>
#include <algorithm>
#include <iomanip>

void sort_coo(size_t *row_idx, size_t *col_idx, double *values, size_t nonzeroelem);
double* pre_filled_array(size_t size, double value);
std::tuple<size_t*, size_t*, double*, size_t, size_t, size_t> get_COO(const char* file_name);
void array_to_stdout(double *array, size_t rows);
__global__ void coo_spmv_kernel(const size_t *row, const size_t *col, const double *val, const double *arr, double *res, size_t nonzeroelem);

int main(int argc, char** argv) {
  if(argc != 2){
	std::cout << "Usage output.exec <matrix-file>" << std::endl;
	return 1;
  }
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  auto [row_indices, col_indices, vals, rows, cols, non_zero_count] = get_COO(argv[1]);
  double* array1 = pre_filled_array(cols, 1.0f);
  double* resulting_array = pre_filled_array(rows, 0);
  //sort_coo(row_indices, col_indices, vals, non_zero_count);

  size_t blockSize = 256;
  size_t gridSize = (non_zero_count + blockSize - 1) / blockSize;

  cudaEventRecord(start);
  coo_spmv_kernel<<<gridSize, blockSize>>>(row_indices, col_indices, vals, array1, resulting_array, non_zero_count);
  cudaEventRecord(stop);
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel Time: %f ms\n", milliseconds);
  //array_to_stdout(resulting_array, rows);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(row_indices);
  cudaFree(col_indices);
  cudaFree(vals);
  cudaFree(array1);
  cudaFree(resulting_array);
  return 0;
}

__global__ void coo_spmv_kernel(const size_t *row, const size_t *col, const double *val, const double *arr, double *res, size_t nonzeroelem) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	// if (i < 5) printf("Thread %lu, row=%lu, col=%lu, val=%.2lf, arr[col]=%.2lf\n", i, row[i], col[i], val[i], arr[col[i]]);
    if (i < nonzeroelem) {
        size_t r = row[i];
        size_t c = col[i];
        double v = val[i];
        atomicAdd(&res[r], v * arr[c]);
    }
}

void sort_coo(size_t *row_idx, size_t *col_idx, double *values, size_t nonzeroelem) {
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
        values[i]  = sorted_val[i];
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
