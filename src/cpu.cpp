#include "include/cpu.h"

#include <iostream>

void spmv_cpu(const IDX_TYPE *row_indices, const IDX_TYPE *col_indices,
			  const NUM_TYPE *val, const NUM_TYPE *arr, const size_t nnz,
			  NUM_TYPE* resulting_array) {
  for(size_t COO_index = 0; COO_index < nnz; ++COO_index){
	resulting_array[row_indices[COO_index]] += (val[COO_index] * arr[row_indices[COO_index]]);
  }
}
