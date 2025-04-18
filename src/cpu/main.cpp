#include "parser.h"
#include <cstddef>
#include <iostream>

double* pre_filled_array(size_t size, double value);

int main(void) {
  auto [row_indices, col_indices, vals, rows, cols, non_zero_count] = get_COO("../dataset/bcsstm09.mtx");
  double* array1 = pre_filled_array(cols, 1.0f);
  double* resulting_array = pre_filled_array(rows, 0);
  
  for(size_t COO_index = 0; COO_index < non_zero_count; ++COO_index){
	resulting_array[row_indices[COO_index]] += (vals[COO_index] * array1[row_indices[COO_index]]);
  }
  
  delete [] row_indices;
  delete [] col_indices;
  delete [] vals;
  delete [] array1;
  delete [] resulting_array;
  return 0;
}

double *pre_filled_array(size_t size, double value) {
  double* res = new double[size];
  for(size_t i = 0; i < size; ++i){
	res[i] = value;
  }
  return res;
}
