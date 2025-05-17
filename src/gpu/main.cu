#include <fstream>
#include <cstdio>
#include <cstring>
#include <cstddef>
#include <tuple>
#include <filesystem>

#define BLOCK_SIZE 512
#define IDX_TYPE size_t
#define NUM_TYPE float
#define NUM_TEST 10
#define OPS_PER_NUN 2
#define MEMEORY_RW 5
// CUDA error checking
#define cudaCheckError(ans)												\
  {																		\
    gpuAssert((ans), __FILE__, __LINE__);								\
  }

typedef void (*gpu_kernel)(const IDX_TYPE *, const IDX_TYPE *, const NUM_TYPE *,
                           const NUM_TYPE *, NUM_TYPE *, const size_t);

inline void gpuAssert(cudaError_t code, const char *file, int line);
NUM_TYPE *pre_filled_array(size_t size, NUM_TYPE value);
std::tuple<IDX_TYPE *, IDX_TYPE *, NUM_TYPE *, IDX_TYPE, IDX_TYPE, IDX_TYPE>
get_COO(const char *filename);
double flops_counter(size_t nnz, float ms);
double mu_fn(double* v, size_t n);
double sigma_fn(double* v, double mu, size_t n);

void test_spmv(gpu_kernel kernel, const IDX_TYPE *row_indices,
               const IDX_TYPE *col_indices, const NUM_TYPE *val,
               const NUM_TYPE *arr, const IDX_TYPE num_rows, const size_t nnz);

__global__ void spmv_without_striding(const IDX_TYPE *row, const IDX_TYPE *col,
                                      const NUM_TYPE *val, const NUM_TYPE *arr,
                                      NUM_TYPE *res, const size_t nnz);

__global__ void spmv_with_striding(const IDX_TYPE *row, const IDX_TYPE *col,
                                   const NUM_TYPE *val, const NUM_TYPE *arr,
                                   NUM_TYPE *res, size_t nnz);

int main(int argc, char **argv) {

  if (argc != 2) {
    fprintf(stderr, "Wrong number of arguments\n");
    fprintf(stderr, "Usage output.exec <matrix-file-path>\n");
    exit(1);
  }

  if (!std::filesystem::exists(argv[1])) {
    fprintf(stderr, "File does not exist\n");
    exit(2);
  }

  auto [row_indices, col_indices, vals, nrows, ncols, nnz] = get_COO(argv[1]);

  NUM_TYPE *array_to_mul = pre_filled_array(ncols, 1.0f);

  printf("###Kernel with striding###\n");
  test_spmv(spmv_with_striding, row_indices, col_indices, vals, array_to_mul,
            nrows, nnz);

  printf("###Kernel with sequantial access###\n");
  test_spmv(spmv_without_striding, row_indices, col_indices, vals, array_to_mul,
            nrows, nnz);

  cudaCheckError(cudaFree(row_indices));
  cudaCheckError(cudaFree(col_indices));
  cudaCheckError(cudaFree(vals));
  cudaCheckError(cudaFree(array_to_mul));
  return 0;
}

double flops_counter(size_t nnz, float ms) {
  size_t flops = OPS_PER_NUN * nnz;
  return (flops / (ms / 1.e3)) / 1.e9;
}

__global__ void spmv_with_striding(const IDX_TYPE *row, const IDX_TYPE *col,
                                    const NUM_TYPE *val, const NUM_TYPE *arr,
                                    NUM_TYPE *res, const size_t nnz) {
  size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t n_thread = blockDim.x * gridDim.x;
  
  IDX_TYPE start = thread_idx;
  IDX_TYPE end = nnz;

  for(IDX_TYPE i = start; i < end; i += n_thread){
	IDX_TYPE row_idx = row[i];
	atomicAdd(&res[row_idx], val[row_idx] * arr[col[i]]);
  }
}

__global__ void spmv_without_striding(const IDX_TYPE *row, const IDX_TYPE *col,
                                 const NUM_TYPE *val, const NUM_TYPE *arr,
                                 NUM_TYPE *res, const size_t nnz) {
  size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t n_thread = blockDim.x * gridDim.x;
  float frac_portion = (float)nnz / (float)n_thread;

  // Ceil function
  size_t portion = ((size_t)frac_portion) + (((size_t)frac_portion) < frac_portion);
  
  IDX_TYPE start = portion * thread_idx;
  IDX_TYPE end = (nnz < portion * (thread_idx + 1))? nnz : portion * (thread_idx + 1);
  
  for(IDX_TYPE i = start; i < end; i++){
	IDX_TYPE row_idx = row[i];
	atomicAdd(&res[row_idx], val[row_idx] * arr[col[i]]);
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

void test_spmv(gpu_kernel kernel, const IDX_TYPE *row_indices,
               const IDX_TYPE *col_indices, const NUM_TYPE *val,
               const NUM_TYPE *arr, const IDX_TYPE num_rows, const size_t nnz) {
  double times[NUM_TEST];
  double flops[NUM_TEST];
  double bandwidth[NUM_TEST];

  cudaEvent_t start, stop;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));

  NUM_TYPE *resulting_array = pre_filled_array(num_rows, 0);

  size_t gridSize = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;

  printf("Number of thread invocked %lu for a total of %lu nnz\n", BLOCK_SIZE * gridSize, nnz);
  
  for (size_t i = 0; i < NUM_TEST; ++i) {

    cudaCheckError(cudaEventRecord(start));
    kernel<<<gridSize, BLOCK_SIZE>>>(row_indices, col_indices, val, arr, resulting_array, nnz);
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));
    //cudaCheckError(cudaDeviceSynchronize());

    float milliseconds = 0;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));

    flops[i] = flops_counter(nnz, milliseconds);
    times[i] = milliseconds;
	bandwidth[i] = (nnz * sizeof(NUM_TYPE) * MEMEORY_RW / milliseconds) / 1e12;
	
	printf("Run %lu developed %lf GFLOP/s in %lf ms with a dandwidth of %lf GB/s against a limit of 933 GB/s\n", i, flops[i], times[i], bandwidth[i]);
	
  }
  
  for(size_t i = 0; i < num_rows; ++i){
	printf("result[%lu] = %lf\n", i, resulting_array[i]);
  }

  cudaCheckError(cudaFree(resulting_array));
  
  double flops_mu = mu_fn(flops, NUM_TEST);
  double flops_sigma = sigma_fn(flops, flops_mu, NUM_TEST);
  
  printf("This kernel produced an average of %lf GFLOP/s with std.dev. of %lf GFLOP/s\n", flops_mu, flops_sigma);

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

inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}
