#include "../distributed_mmio/include/mmio_c_wrapper.h"

#include "../include/colours.h"
#include "../include/cli.hpp"
#include "../include/testers.cuh"

int main(int argc, char **argv) {
  Cli_Args args;
  init_cli();
  if (parse_args(argc, argv, &args) != 0) {
    return -1;
  }

  mmio_coo_u64_f32_t *coo_matrix = mmio_read_coo_u64_f32(args.filename, true);

  printf("The matrix has %lu and the last value is %f\n",coo_matrix -> nnz, coo_matrix -> val[coo_matrix -> nnz - 1]);

  printf(RED "Now running the baseline" RESET "\n");
  test_spmv(coo_matrix, kernel_type::BASELINE);

  printf(RED "Now running the warp shuffle" RESET "\n");
  test_spmv(coo_matrix, kernel_type::WARP_SHFL);

  printf(RED "Now running the warp shuffle with unroll" RESET "\n");
  test_spmv(coo_matrix, kernel_type::WARP_SHFL_UNROLL);

  mmio_destroy_coo_u64_f32(coo_matrix);
  return 0;
}
