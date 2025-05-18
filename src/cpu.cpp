#include "include/cpu.h"

#include <iostream>

void spmv_cpu(const IDX_TYPE *row_indices, const IDX_TYPE *col_indices,
			  const NUM_TYPE *val, const NUM_TYPE *arr, const size_t nnz,
			  NUM_TYPE* resulting_array) {
  //printf("I primi tre valori sono %lf, %lf, %lf\n",val[0], val[1], val[2]);
  //printf("I primi tre valori del risultato sono %lf, %lf, %lf\n",resulting_array[0], resulting_array[1], resulting_array[2]);
  for(size_t COO_index = 0; COO_index < nnz; ++COO_index){
	resulting_array[row_indices[COO_index]] += (val[COO_index] * arr[row_indices[COO_index]]);
  }
}
