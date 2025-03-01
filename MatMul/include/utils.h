#ifndef UTILITY_H
#define UTILITY_H
#include <cstdio>

#define B 1
#define M 1024
#define N 512
#define K 12000

template <typename T>
void dataShow(T *mat) {
    printf("output head && tail:\n");

    printf("%-14.6f  %-14.6f  %-14.6f  %-14.6f  %-14.6f  %-14.6f  %-14.6f  %-14.6f\n",
           mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7]);

    printf("%-14.6f  %-14.6f  %-14.6f  %-14.6f  %-14.6f  %-14.6f  %-14.6f  %-14.6f\n",
           mat[B*M*N-8], mat[B*M*N-7], mat[B*M*N-6], mat[B*M*N-5], 
           mat[B*M*N-4], mat[B*M*N-3], mat[B*M*N-2], mat[B*M*N-1]);

    printf("\n");
}

#endif // UTILITY_H