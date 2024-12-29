#ifndef MATMUL_H
#define MATMUL_H

#include "utils.h"

int matmul(float *matA, float *matB, float *matC);
int matmulN(float *matA, float *matB, float *matC);
int matmulMN(float *matA, float *matB, float *matC);
int matmulMNK(float *matA, float *matB, float *matC);
int matmulMNKOpt(float *matA, float *matB, float *matC);

#endif  // MATMUL_H
