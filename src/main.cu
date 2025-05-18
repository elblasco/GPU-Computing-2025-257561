#include <filesystem>

#include "include/gpu.h"
#include "include/testers.h"
#include "include/utils.h"

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

  NUM_TYPE *array_to_mul_cpu = pre_filled_array_cpu(ncols, 1.0f);
  NUM_TYPE *array_to_mul_gpu = pre_filled_array_gpu(ncols, 1.0f);

  printf("###### CPU SPMV ######\n");
  NUM_TYPE *cpu_result = test_spmv_cpu(row_indices, col_indices, vals,
                                       array_to_mul_cpu, nrows, nnz);

  printf("###### GPU Kernel with striding ######\n");
  NUM_TYPE *strinding_result =
      test_spmv_gpu(spmv_with_striding, row_indices, col_indices, vals,
                    array_to_mul_gpu, nrows, nnz);

  printf("###### GPU Kernel with sequantial access ######\n");
  NUM_TYPE *sequantial_result =
      test_spmv_gpu(spmv_without_striding, row_indices, col_indices, vals,
                    array_to_mul_gpu, nrows, nnz);

  delete[] cpu_result;
  delete[] array_to_mul_cpu;
  cudaCheckError(cudaFree(array_to_mul_gpu));
  cudaCheckError(cudaFree(strinding_result));
  cudaCheckError(cudaFree(sequantial_result));

  return 0;
}
