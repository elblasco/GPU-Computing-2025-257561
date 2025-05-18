#include "include/utils.h"
#include "include/macros.h"

NUM_TYPE *pre_filled_array_cpu(size_t size, NUM_TYPE value) {
  NUM_TYPE* res = new NUM_TYPE[size];
  for(size_t i = 0; i < size; ++i){
	res[i] = value;
  }
  return res;
}

NUM_TYPE *pre_filled_array_gpu(size_t size, NUM_TYPE value) {
  NUM_TYPE *arr;
  cudaMallocManaged(&arr, size * sizeof(NUM_TYPE));
  for (size_t i = 0; i < size; ++i) {
    arr[i] = value;
  }
  return arr;
}

std::tuple<IDX_TYPE *, IDX_TYPE *, NUM_TYPE *, IDX_TYPE, IDX_TYPE, IDX_TYPE>
get_COO(const char *filename) {
  IDX_TYPE *row_indices, *col_indices;
  NUM_TYPE *vals;
  size_t nrows, ncols, nnz;
  std::ifstream infile(filename);

  std::string line;
  // Skip comments
  while (std::getline(infile, line)) {
    if (line[0] != '%')
      break;
  }

  std::istringstream header(line);
  header >> nrows >> ncols >> nnz;

  cudaCheckError(cudaMallocManaged(&row_indices, nnz * sizeof(IDX_TYPE)));
  cudaCheckError(cudaMallocManaged(&col_indices, nnz * sizeof(IDX_TYPE)));
  cudaCheckError(cudaMallocManaged(&vals, nnz * sizeof(NUM_TYPE)));

  IDX_TYPE row, col;
  NUM_TYPE val;
  for (size_t i = 0; i < nnz; ++i) {
    infile >> row >> col >> val;
    row_indices[i] = col - 1;
    col_indices[i] = row - 1;
    vals[i] = val;
  }

  return {row_indices, col_indices, vals, nrows, ncols, nnz};
}

double mu_fn(double* v, size_t n){
	double sum = 0;
	for (size_t i = 0; i < n; i++){
		sum += v[i];
	}
	return sum / n;
}

double sigma_fn(double* v, double mu, size_t n){
	double sum = 0;
	for (size_t i = 0; i<n; ++i){
		sum += pow(v[i] - mu,2);
	}
	return std::sqrt(sum / n);
}

double flops_counter(size_t nnz, float ms) {
  size_t flops = OPS_PER_NUN * nnz;
  return (flops / (ms / 1.e3)) / 1.e9;
}
