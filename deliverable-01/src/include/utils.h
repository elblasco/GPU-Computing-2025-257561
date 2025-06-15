#pragma once

#include <cstdlib>
#include <tuple>
#include <fstream>
#include <sstream>
#include <cmath>
#include "macros.h"

NUM_TYPE *pre_filled_array_cpu(size_t size, NUM_TYPE value);

NUM_TYPE *pre_filled_array_gpu(size_t size, NUM_TYPE value);

std::tuple<IDX_TYPE*, IDX_TYPE*, NUM_TYPE*, IDX_TYPE, IDX_TYPE, IDX_TYPE> get_COO(const char* file_name);

double mu_fn(double *v, size_t n);

double sigma_fn(double* v, double mu, size_t n);

double flops_counter(size_t nnz, float ms);
