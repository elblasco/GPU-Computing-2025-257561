#include <cstddef>
#include <iostream>
#include <fstream>
#include <tuple>
#include <sstream>
#include <cmath>
#include <chrono>
#include <ctime>

#define IDX_TYPE size_t
#define NUM_TYPE float
#define NUM_TEST 10

NUM_TYPE* pre_filled_array(size_t size, NUM_TYPE value);
std::tuple<IDX_TYPE*, IDX_TYPE*, NUM_TYPE*, IDX_TYPE, IDX_TYPE, IDX_TYPE> get_COO(const char* file_name);
double flops_counter(size_t nnz, float ms);
double mu_fn(double* v, size_t n);
double sigma_fn(double* v, double mu, size_t n);
void test_spmv_cpu(const IDX_TYPE *row_indices,
				   const IDX_TYPE *col_indices, const NUM_TYPE *val,
				   const NUM_TYPE *arr, const IDX_TYPE num_rows,
				   const NUM_TYPE *test_res, const size_t nnz);

int main(int argc, char** argv) {

  if(argc != 2){
	std::cout << "Usage output.exec <matrix-file>" << std::endl;
	return 1;
  }
  
  auto [row_indices, col_indices, vals, rows, cols, non_zero_count] = get_COO(argv[1]);

  NUM_TYPE* array1 = pre_filled_array(cols, 1.0f);
  NUM_TYPE* resulting_array = pre_filled_array(rows, 0);
  
  test_spmv_cpu(row_indices, col_indices, vals,array1, rows, resulting_array, non_zero_count);
  
  delete [] row_indices;
  delete [] col_indices;
  delete [] vals;
  delete [] array1;
  delete [] resulting_array;
  return 0;
}

NUM_TYPE *pre_filled_array(size_t size, NUM_TYPE value) {
  NUM_TYPE* res = new NUM_TYPE[size];
  for(size_t i = 0; i < size; ++i){
	res[i] = value;
  }
  return res;
}

std::tuple<IDX_TYPE*, IDX_TYPE*, NUM_TYPE*, IDX_TYPE, IDX_TYPE, IDX_TYPE> get_COO(const char* file_name) {
  std::ifstream MyFile(file_name);
  std::string line;
  // Skip header/comments
  do {
	std::getline(MyFile, line);
  } while (line[0] == '%');

  IDX_TYPE rows, cols, nonzero_vals;
  IDX_TYPE* row_indices;
  IDX_TYPE* col_indices;
  NUM_TYPE* vals;
  std::istringstream sizes(line);
  sizes >> rows >> cols >> nonzero_vals;

  row_indices = new IDX_TYPE[nonzero_vals];
  col_indices = new IDX_TYPE[nonzero_vals];
  vals = new NUM_TYPE[nonzero_vals];

  IDX_TYPE row, col;
  NUM_TYPE val;
  for (size_t i = 0; i < nonzero_vals; ++i) {
    MyFile >> row >> col >> val;
    row_indices[i] = col - 1;
    col_indices[i] = row - 1;
    vals[i] = val;
  }
  
  MyFile.close();
  return {row_indices, col_indices, vals, rows, cols, nonzero_vals};
}

double mu_fn(double* v, size_t n){
	double sum = 0;
	for (size_t i = 0; i < n; i++){
		sum += v[i];
	}
	return sum / n;
}

double sigma_fn(double* v, double mu, size_t n){
	long double sum = 0;
	for (size_t i = 0; i<n; ++i){
		sum += pow(v[i] - mu,2);
	}
	return sum / n;
}

double flops_counter(size_t nnz, float ms) {
  size_t flops = 2 * nnz;
  return (flops / (ms / 1.e3)) / 1.e19;
}

void test_spmv_cpu(const IDX_TYPE *row_indices,
                           const IDX_TYPE *col_indices, const NUM_TYPE *val,
                           const NUM_TYPE *arr, const IDX_TYPE num_rows,
                           const NUM_TYPE *test_res, const size_t nnz) {
  printf("###### 1 thread per 1 element kernel ######\n");
  double times[NUM_TEST];
  double flops[NUM_TEST];
  NUM_TYPE *resulting_array = pre_filled_array(num_rows, 1);
  
  for (size_t i = 0; i < NUM_TEST; ++i) {
	
    auto start = std::chrono::system_clock::now();
	
	for(size_t COO_index = 0; COO_index < nnz; ++COO_index){
	  resulting_array[row_indices[COO_index]] += (val[COO_index] * arr[row_indices[COO_index]]);
	  printf("result[%lu] = %f", COO_index, resulting_array[row_indices[COO_index]]);
	  fflush(stdout);
	}
	
	auto end = std::chrono::system_clock::now();
	
    double elapsed_ms = (end-start).count() * 1e3;
	flops[i] = flops_counter(nnz, elapsed_ms);
	times[i] = elapsed_ms;
  }
  delete[] resulting_array;
  double flops_mu = mu_fn(flops, NUM_TEST);
  double flops_sigma = sigma_fn(flops, flops_mu, NUM_TEST);
  
  printf("This kernel produced an average of %lf GFLOPS with std.dev. of %lf GFLOPS\n", flops_mu, flops_sigma);

  double times_mu = mu_fn(times, NUM_TEST);
  double times_sigma = sigma_fn(times, times_mu, NUM_TEST);
  
  printf("This kernel executed with an average of %lf ms with std.dev. of %lf ms\n", flops_mu, times_sigma);
}
