#include <cstddef>
#include <iostream>
#include <fstream>
#include <tuple>
#include <sstream>

double* pre_filled_array(size_t size, double value);
std::tuple<size_t*, size_t*, double*, size_t, size_t, size_t> get_COO(const char* file_name);
void array_to_file(const char *file_name, double *array, size_t rows);

int main(int argc, char** argv) {
  if(argc != 3){
	std::cout << "Usage output.exec <matrix-file>" << std::endl;
	return 1;
  }
  auto [row_indices, col_indices, vals, rows, cols, non_zero_count] = get_COO(argv[1]);
  double* array1 = pre_filled_array(cols, 1.0f);
  double* resulting_array = pre_filled_array(rows, 0);
  
  for(size_t COO_index = 0; COO_index < non_zero_count; ++COO_index){
	resulting_array[row_indices[COO_index]] += (vals[COO_index] * array1[row_indices[COO_index]]);
  }

  array_to_file(argv[2], resulting_array, rows);
  
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

std::tuple<size_t*, size_t*, double*, size_t, size_t, size_t> get_COO(const char* file_name) {
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

void array_to_file(const char *file_name, double* array, size_t rows) {
  std::ofstream out_file(file_name);
  if (!out_file) {
        std::cerr << "Error opening file: " << file_name << std::endl;
        return;
  }
  
  out_file << "This is the encoding of the result\n";
  out_file << "The first line is the number of rows\n";
  out_file << "Each line is a cell of the array\n";
  
  out_file << rows << "\n";
  
  for (size_t row = 0; row < rows; ++row) {
	out_file << array[row] << "\n";
  }
}
