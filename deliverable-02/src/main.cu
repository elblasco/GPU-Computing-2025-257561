#include "../distributed_mmio/include/mmio.h"
#include "../distributed_mmio/include/mmio_utils.h"

#include "../include/colours.h"
#include "../include/utils.cuh"
#include "../include/testers.cuh"
#include "../include/gpu.cuh"
#include "../include/cli.hpp"

int main(int argc, char **argv) {
  Cli_Args args;
  init_cli();
  if (parse_args(argc, argv, &args) != 0) {
    return -1;
  }

  printf("Working on matrix: %s\n", args.filename);
  
  COO_local<IDX_TYPE, NUM_TYPE> *coo_matrix = Distr_MMIO_COO_local_read<IDX_TYPE, NUM_TYPE>(args.filename);

  test_spmv_gpu(spmv_with_striding, coo_matrix);

  test_spmv_gpu(spmv_without_striding, coo_matrix);

  return 0;
}
