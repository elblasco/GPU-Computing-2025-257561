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

  printf(GREEN "Now running the baseline" RESET "\n");
  test_spmv(coo_matrix, kernel_type::BASELINE);

  printf(GREEN "Now running the warp shuffle" RESET "\n");
  test_spmv(coo_matrix, kernel_type::WARP_SHFL);

  printf(GREEN "Now running the warp shuffle with unroll" RESET "\n");
  test_spmv(coo_matrix, kernel_type::WARP_SHFL_UNROLL);

  printf(GREEN "Now running the shared memory " RESET "\n");
  test_spmv(coo_matrix, kernel_type::SHARED_MEMORY_SUM);

  return 0;
}
