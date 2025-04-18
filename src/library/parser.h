#pragma once

#include <cstddef>
#include <fstream>
#include <iostream>
#include <tuple>
#include <sstream>

std::tuple<size_t*, size_t*, double*, size_t, size_t, size_t> get_COO(char* file_name) {
  std::ifstream MyFile(file_name);
  std::string line;
  // Skip header/comments
  do {
	std::getline(MyFile, line);
  } while (line[0] == '%');

  size_t rows, cols, nonzero_vals;
  size_t* row_indices;
  size_t* col_indices;
  double* vals;
  std::istringstream sizes(line);
  sizes >> rows >> cols >> nonzero_vals;

  row_indices = new size_t[nonzero_vals];
  col_indices = new size_t[nonzero_vals];
  vals = new double[nonzero_vals];
  
  for(size_t i = 0; i < nonzero_vals; ++i){
    size_t row, col;
	double val;
	MyFile >> row;
	MyFile >> col;
	MyFile >> val;
	row_indices[i] = row - 1;
	col_indices[i] = col - 1;
	vals[i] = val;
  }
  
  MyFile.close();
  return {row_indices, col_indices, vals, rows, cols, nonzero_vals};
}
