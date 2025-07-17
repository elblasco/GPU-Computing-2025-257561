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

  COO_local<IDX_TYPE, NUM_TYPE> *coo_matrix =
      Distr_MMIO_sorted_COO_local_read<IDX_TYPE, NUM_TYPE>(args.filename,
                                                           false);

  printf(RED "Now running the baseline" RESET "\n");
  test_spmv(coo_matrix, kernel_type::BASELINE);

  printf(RED "Now running the warp shuffle" RESET "\n");
  test_spmv(coo_matrix, kernel_type::WARP_SHFL);

  printf(RED "Now running the warp shuffle with unroll" RESET "\n");
  test_spmv(coo_matrix, kernel_type::WARP_SHFL_UNROLL);

  // mmio_destroy_coo_u64_f32(coo_matrix);
  return 0;
}
