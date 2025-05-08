#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <vector>

#define NROWS 10000
#define NCOLS 10000
#define NNZ 500
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

double *pre_filled_array(size_t size, double value);
std::tuple<size_t *, size_t *, double *> get_COO();
void build_row_ptr_from_coo(size_t *row_idx, size_t *&d_row_offsets_out);
void flops_counter(size_t nnz, float ms);

__global__ void entire_row(const size_t *row, const size_t *col,
                           const double *val, const double *arr, double *res,
                           const size_t num_rows, const size_t *row_offsets);

void test_entire_row(const size_t *row_indices, const size_t *col_indices,
                     const double *val, const double *arr,
                     const size_t num_rows, const size_t *row_offsets,
                     const double *test_res, const size_t nnz);

__global__ void test_coo_spmv_kernel(const size_t *row, const size_t *col,
                                     const double *val, const double *arr,
                                     double *res, size_t nonzeroelem);

int main(void) {

  auto [row_indices, col_indices, vals] = get_COO();

  double *array_to_mul = pre_filled_array(NCOLS, 1.0f);
  double *cmp_resulting_array = pre_filled_array(NROWS, 0);
  size_t *row_offsets;
  build_row_ptr_from_coo(row_indices, row_offsets);

  size_t gridSizeNNZ = (NNZ + BLOCK_SIZE - 1) / BLOCK_SIZE;

  test_coo_spmv_kernel<<<gridSizeNNZ, BLOCK_SIZE>>>(
      row_indices, col_indices, vals, array_to_mul, cmp_resulting_array, NNZ);

  cudaDeviceSynchronize();

  test_entire_row(row_indices, col_indices, vals, array_to_mul, NROWS,
                  row_offsets, cmp_resulting_array, NNZ);

  cudaFree(row_offsets);
  cudaFree(row_indices);
  cudaFree(col_indices);
  cudaFree(vals);
  cudaFree(array_to_mul);
  cudaFree(cmp_resulting_array);
  return 0;
}

void flops_counter(size_t nnz, float ms) {
  size_t flops = 2 * nnz;
  // printf("We have a total of %lu FP64 operations\n", flops);
  printf("We developed a total of %lf TFLOPs\n", (flops / (ms / 1.e3)) / 1.e12);
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

void build_row_ptr_from_coo(size_t *row_idx, // COO row indices [nnz]
                            size_t *&d_row_offsets_out) {

  std::vector<size_t> row_offsets_host(NROWS + 1, 0);
  // Step 1: Count number of nonzeros per row
  for (size_t i = 0; i < NNZ; ++i)
    row_offsets_host[row_idx[i] + 1]++;

  // Step 2: Cumulative sum to get row_ptr
  for (size_t i = 0; i <= NROWS; ++i)
    row_offsets_host[i + 1] += row_offsets_host[i];

  cudaMalloc(&d_row_offsets_out, (NROWS + 1) * sizeof(size_t));
  cudaMemcpy(d_row_offsets_out, row_offsets_host.data(),
             (NROWS + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
}

std::tuple<size_t *, size_t *, double *> get_COO() {
  fprintf(stderr, "Sono in get COO\n");
  fflush(stdout);
  
  srand((unsigned int)time(NULL));

  fprintf(stderr, "Sono in get COO e ho istanziato il rng\n");
  fflush(stdout);
  
  size_t count = 0;
  size_t *row_indices, *col_indices;
  double *vals;

  cudaMallocManaged(&row_indices, NNZ * sizeof(size_t));
  cudaMallocManaged(&col_indices, NNZ * sizeof(size_t));
  cudaMallocManaged(&vals, NNZ * sizeof(double));
  
  // Track used positions (row-major indexing)
  int used[NROWS * NCOLS] = {0};

  fprintf(stderr, "Sono in get COO e ho dichiarato tutto\n");
  fflush(stdout);
  
  for (int r = 0; r < NROWS && count < NNZ; ++r) {
	for (int c = 0; c < NCOLS && count < NNZ; ++c) {
	  int index = r * NCOLS + c;
	  if (!used[index] && (rand() % 2)) { // Randomly decide to place a non-zero
		used[index] = 1;
		row_indices[count] = r;
		col_indices[count] = c;
		vals[count] = (double)(rand() % 100 + 1);
		count++;
	  }
	}
  }

  for (int r = 0; r < NROWS && count < NNZ; ++r) {
    for (int c = 0; c < NCOLS && count < NNZ; ++c) {
      int index = r * NCOLS + c;
      if (!used[index]) {
        used[index] = 1;
        row_indices[count] = r;
        col_indices[count] = c;
        vals[count] = (double)(rand() % 100 + 1);
        count++;
      }
    }
  }

  fprintf(stderr, "Sono in get COO e ho inizializzato tutto\n");
  fflush(stdout);

  return {row_indices, col_indices, vals};
}

double *pre_filled_array(size_t size, double value) {
  double* arr;
  cudaMallocManaged(&arr, size * sizeof(double));
  for (size_t i = 0; i < size; ++i) {
    arr[i] = value;
  }
  printf("Prefilled array done\n");
  return arr;
}

void print_double_bits(double value) {
  union {
    double d;
    uint64_t u;
  } converter;

  converter.d = value;

  for (int i = 63; i >= 0; --i) {

    printf("%d", (converter.u >> i) & 1);
    if (i % 4 == 0)
      printf(" "); // optional: space every 4 bits
  }
  printf("(%.15f)\n", value);
}

void test_entire_row(const size_t *row_indices, const size_t *col_indices,
                     const double *val, const double *arr,
                     const size_t num_rows, const size_t *row_offsets,
                     const double *test_res, const size_t nnz) {

  // for(size_t i = 0; i < num_rows; ++i)
  // printf("%lu ", d_row_offsets[i]);

  printf("I am launching the one thread one row kernel\n");
  printf("It should uses %lu threads\n", num_rows);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double *d_resulting_array = pre_filled_array(num_rows, 0);
  double *h_resulting_array = NULL, *h_test_res = NULL;

  size_t gridSizeRowNoStride = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  printf("The number of thread invoked launching the kernle is %lu\n",
         BLOCK_SIZE * gridSizeRowNoStride);
  cudaEventRecord(start);
  entire_row<<<gridSizeRowNoStride, BLOCK_SIZE>>>(row_indices, col_indices, val,
                                                  arr, d_resulting_array,
                                                  num_rows, row_offsets);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel which takes a row for each thread Time: %f ms\n", milliseconds);
  
  cudaDeviceSynchronize();
  
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
  
  cudaFree(d_resulting_array);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
