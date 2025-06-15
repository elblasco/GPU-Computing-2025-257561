#pragma once

#include <cstdlib>
#include "macros.h"

void spmv_cpu(const IDX_TYPE *row_indices, const IDX_TYPE *col_indices,
			  const NUM_TYPE *val, const NUM_TYPE *arr, const size_t nnz,
			  NUM_TYPE *resulting_array);
