#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <filesystem>

#define BLOCK_SIZE 512
#define IDX_TYPE size_t
#define NUM_TYPE float
#define NUM_TEST 10

// CUDA error checking
#define cudaCheckError(ans)                                                    \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

NUM_TYPE *pre_filled_array(size_t size, NUM_TYPE value);
std::tuple<IDX_TYPE *, IDX_TYPE *, NUM_TYPE *, IDX_TYPE, IDX_TYPE, IDX_TYPE>
get_COO(const char *filename);
double flops_counter(size_t nnz, float ms);
double mu_fn(double* v, size_t n);
double sigma_fn(double* v, double mu, size_t n);

__global__ void spmv_per_entire_row(const IDX_TYPE *row, const IDX_TYPE *col,
                           const NUM_TYPE *val, const NUM_TYPE *arr,
                           NUM_TYPE *res, const size_t num_rows,
                           const size_t nnz);

void test_spmv_per_entire_row(const IDX_TYPE *row_indices, const IDX_TYPE *col_indices,
                     const NUM_TYPE *val, const NUM_TYPE *arr,
                     const IDX_TYPE num_rows, const NUM_TYPE *test_res,
                     const size_t nnz);

__global__ void spmv_with_atomic(const IDX_TYPE *row, const IDX_TYPE *col,
                                     const NUM_TYPE *val, const NUM_TYPE
                                     *arr, NUM_TYPE *res, size_t
                                     nonzeroelem);

void test_spmv_with_atomic(const IDX_TYPE *row_indices, const IDX_TYPE *col_indices,
                     const NUM_TYPE *val, const NUM_TYPE *arr,
                     const IDX_TYPE num_rows, const NUM_TYPE *test_res,
                     const size_t nnz);

int main(int argc, char **argv) {

  if(argc != 2){
	fprintf(stderr, "Wrong number of arguments\n");
	fprintf(stderr, "Usage output.exec <matrix-file-path>\n");
	exit(1);
  }

  if(!std::filesystem::exists(argv[1])){
	fprintf(stderr, "File does not exist\n");
	exit(2);
  }
  
  auto [row_indices, col_indices, vals, nrows, ncols, nnz] = get_COO(argv[1]);

  NUM_TYPE *array_to_mul = pre_filled_array(ncols, 1.0f);
  NUM_TYPE *cmp_resulting_array = pre_filled_array(nrows, 0);

  test_spmv_with_atomic(row_indices, col_indices, vals, array_to_mul, nrows,
                  cmp_resulting_array, nnz);
  
  test_spmv_per_entire_row(row_indices, col_indices, vals, array_to_mul, nrows,
                  cmp_resulting_array, nnz);

  cudaCheckError(cudaFree(row_indices));
  cudaCheckError(cudaFree(col_indices));
  cudaCheckError(cudaFree(vals));
  cudaCheckError(cudaFree(array_to_mul));
  cudaCheckError(cudaFree(cmp_resulting_array));
  return 0;
}

double flops_counter(size_t nnz, float ms) {
  size_t flops = 2 * nnz;
  return (flops / (ms / 1.e3)) / 1.e12;
}

__global__ void spmv_per_entire_row(const IDX_TYPE *row, const IDX_TYPE *col,
                           const NUM_TYPE *val, const NUM_TYPE *arr,
                           NUM_TYPE *res, const size_t num_rows,
                           const size_t nnz) {
  size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_idx >= num_rows)
    return;

  IDX_TYPE start = 0;
  while (start < nnz && row[start] < thread_idx)
    ++start;

  IDX_TYPE end = start;
  while (end < nnz && row[end] == thread_idx)
    ++end;

  // printf("Thread number %lu should do from index %lu to index %lu\n",
  // thread_idx, start, end);

  NUM_TYPE sum = 0;

  for (size_t i = start; i < end; ++i) {
    sum += val[i] * arr[col[i]];
  }

  res[thread_idx] = sum;
}

__global__ void spmv_with_atomic(const IDX_TYPE *row, const IDX_TYPE *col,
                                     const NUM_TYPE *val, const NUM_TYPE
                                     *arr, NUM_TYPE *res, size_t nonzeroelem)
                                     {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nonzeroelem) {
    size_t r = row[i];
    size_t c = col[i];
    double v = val[i];
    atomicAdd(&res[r], v * arr[c]);
  }
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

NUM_TYPE *pre_filled_array(size_t size, NUM_TYPE value) {
  NUM_TYPE *arr;
  cudaMallocManaged(&arr, size * sizeof(NUM_TYPE));
  for (size_t i = 0; i < size; ++i) {
    arr[i] = value;
  }
  return arr;
}

void print_bits(NUM_TYPE value) {
  union {
    NUM_TYPE d;
    uint64_t u;
  } converter;

  converter.d = value;

  for (int i = 63; i >= 0; --i) {
    printf("%d", (converter.u >> i) & 1);
    if (i % 4 == 0)
      printf(" "); // optional: space every 4 bits
  }
  printf("(%lf)\n", value);
}

void test_spmv_per_entire_row(const IDX_TYPE *row_indices, const IDX_TYPE *col_indices,
                     const NUM_TYPE *val, const NUM_TYPE *arr,
                     const IDX_TYPE num_rows, const NUM_TYPE *test_res,
                     const size_t nnz) {
  printf("###### 1 thread per 1 row kernel ######\n");
  double times[NUM_TEST];
  double flops[NUM_TEST];

  cudaEvent_t start, stop;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));

  for (size_t i = 0; i < NUM_TEST; ++i) {
    NUM_TYPE *resulting_array = pre_filled_array(num_rows, 0);
    size_t gridSizeRowNoStride = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

	cudaCheckError(cudaEventRecord(start));
    spmv_per_entire_row<<<gridSizeRowNoStride, BLOCK_SIZE>>>(
        row_indices, col_indices, val, arr, resulting_array, num_rows, nnz);
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));
    cudaCheckError(cudaDeviceSynchronize());
	
    float milliseconds = 0;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));

	flops[i] = flops_counter(nnz, milliseconds);
	times[i] = milliseconds;
	
	cudaCheckError(cudaFree(resulting_array));
  }

  double flops_mu = mu_fn(flops, NUM_TEST);
  double flops_sigma = sigma_fn(flops, flops_mu, NUM_TEST);
  
  printf("This kernel produced an average of %lf FLOPS with std.dev. of %lf FLOPS\n", flops_mu, flops_sigma);

  double times_mu = mu_fn(times, NUM_TEST);
  double times_sigma = sigma_fn(times, times_mu, NUM_TEST);
  
  printf("This kernel executed with an average of %lf ms with std.dev. of %lf ms\n", times_mu, times_sigma);
  
  // size_t errors = 0;
  // for (size_t i = 0; i < num_rows; ++i) {
  // 	if(resulting_array[i] != test_res[i] && std::abs(resulting_array[i] - test_res[i]) > 0.00000001){
  // 	  printf("Error at row %lu\n", i);
  // 	  errors += resulting_array[i] != test_res[i];
  // 	  print_bits(resulting_array[i]);
  // 	  print_bits(test_res[i]);
  // 	}
  // }
  // printf("Errors: %lu\n", errors);
  
  cudaCheckError(cudaEventDestroy(start));
  cudaCheckError(cudaEventDestroy(stop));
}

void test_spmv_with_atomic(const IDX_TYPE *row_indices,
                           const IDX_TYPE *col_indices, const NUM_TYPE *val,
                           const NUM_TYPE *arr, const IDX_TYPE num_rows,
                           const NUM_TYPE *test_res, const size_t nnz) {
  printf("###### 1 thread per 1 element kernel ######\n");
  double times[NUM_TEST];
  double flops[NUM_TEST];

  cudaEvent_t start, stop;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));

  for (size_t i = 0; i < NUM_TEST; ++i) {
    NUM_TYPE *resulting_array = pre_filled_array(num_rows, 0);
	size_t gridSizeNNZ = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;

	cudaCheckError(cudaEventRecord(start));
        spmv_with_atomic<<<gridSizeNNZ, BLOCK_SIZE>>>(
            row_indices, col_indices, val, arr, resulting_array, nnz);
        cudaCheckError(cudaEventRecord(stop));
        cudaCheckError(cudaEventSynchronize(stop));
        cudaCheckError(cudaDeviceSynchronize());

        float milliseconds = 0;
        cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));

	flops[i] = flops_counter(nnz, milliseconds);
	times[i] = milliseconds;
	
	cudaCheckError(cudaFree(resulting_array));
  }

  double flops_mu = mu_fn(flops, NUM_TEST);
  double flops_sigma = sigma_fn(flops, flops_mu, NUM_TEST);
  
  printf("This kernel produced an average of %lf FLOPS with std.dev. of %lf FLOPS\n", flops_mu, flops_sigma);

  double times_mu = mu_fn(times, NUM_TEST);
  double times_sigma = sigma_fn(times, times_mu, NUM_TEST);
  
  printf("This kernel executed with an average of %lf ms with std.dev. of %lf ms\n", times_mu, times_sigma);
  
  cudaCheckError(cudaEventDestroy(start));
  cudaCheckError(cudaEventDestroy(stop));
}

double mu_fn(double* v, size_t n){
	double sum = 0;
	for (size_t i = 0; i < n; i++){
		sum += v[i];
	}
	return sum / n;
}

double sigma_fn(double* v, double mu, size_t n){
	long double sum = 0;
	for (size_t i = 0; i<n; ++i){
		sum += pow(v[i] - mu,2);
	}
	return sum / n;
}
