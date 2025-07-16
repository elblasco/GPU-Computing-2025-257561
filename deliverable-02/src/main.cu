#include "../distributed_mmio/include/mmio.h"

#include "../include/cli.hpp"
#include "../include/colours.h"
#include "../include/testers.cuh"

int main(int argc, char **argv) {
  Cli_Args args;
  init_cli();
  if (parse_args(argc, argv, &args) != 0) {
    return -1;
  }

  // Distr_MMIO_sorted_COO_local_read

  COO_local<IDX_TYPE, NUM_TYPE> *coo_matrix =
      Distr_MMIO_sorted_COO_local_read<IDX_TYPE, NUM_TYPE>(args.filename,
                                                           false);

  // printf("The matrix has %lu and the last value is %f\n",coo_matrix -> nnz,
  // coo_matrix -> val[coo_matrix -> nnz - 1]);

  // printf(RED "Now running the baseline" RESET "\n");
  // test_spmv(coo_matrix, kernel_type::BASELINE);

  // printf(RED "Now running the warp shuffle" RESET "\n");
  // test_spmv(coo_matrix, kernel_type::WARP_SHFL);

  // printf(RED "Now running the warp shuffle with unroll" RESET "\n");
  // test_spmv(coo_matrix, kernel_type::WARP_SHFL_UNROLL);

  for (size_t i = 0; i < coo_matrix->nnz; ++i) {
    printf("%lu %lu\n", coo_matrix->row[i], coo_matrix->col[i]);
  }

  // mmio_destroy_coo_u64_f32(coo_matrix);
  return 0;
}
