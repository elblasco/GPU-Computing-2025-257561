#include "../distributed_mmio/include/mmio.h"
#include "../distributed_mmio/include/mmio_utils.h"

#include "../include/colours.h"
#include "../include/utils.cuh"
#include "../include/cli.hpp"

int main(int argc, char **argv) {
  Cli_Args args;
  init_cli();
  if (parse_args(argc, argv, &args) != 0) {
    return -1;
  }

  CPU_TIMER_INIT(MTX_read);

  COO_local<uint64_t, double> *coo_matrix = Distr_MMIO_COO_local_read<uint64_t, double>(args.filename);

  CPU_TIMER_STOP(MTX_read);

  printf("\n[OUT] MTX file read time: %f ms\n", CPU_TIMER_ELAPSED(MTX_read));

  printf("Matrix size: %.3fM rows, %.3fM nnz\n", coo_matrix->nrows / 1e6, coo_matrix->nnz / 1e6);

  
  return 0;
}
