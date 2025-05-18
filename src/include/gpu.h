#pragma once

#include "macros.h"

__global__ void spmv_without_striding(const IDX_TYPE *row, const IDX_TYPE *col,
                                      const NUM_TYPE *val, const NUM_TYPE *arr,
                                      NUM_TYPE *res, const size_t nnz, const size_t portion);

__global__ void spmv_with_striding(const IDX_TYPE *row, const IDX_TYPE *col,
                                   const NUM_TYPE *val, const NUM_TYPE *arr,
                                   NUM_TYPE *res, size_t nnz, const size_t portion);
