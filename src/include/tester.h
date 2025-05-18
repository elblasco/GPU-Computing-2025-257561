#pragma once

#include "macros.h"

NUM_TYPE *test_spmv_cpu(const IDX_TYPE *row_indices,
                        const IDX_TYPE *col_indices, const NUM_TYPE *val,
                        const NUM_TYPE *arr, const IDX_TYPE num_rows,
                        const size_t nnz);

NUM_TYPE *test_spmv_gpu(gpu_kernel kernel, const IDX_TYPE *row_indices,
                        const IDX_TYPE *col_indices, const NUM_TYPE *val,
                        const NUM_TYPE *arr, const IDX_TYPE num_rows,
                        const size_t nnz);
