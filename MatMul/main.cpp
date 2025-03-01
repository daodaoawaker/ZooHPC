#include <cstdlib>

#include "rng.h"
#include "utils.h"
#include "common.h"
#include "src/matmul.h"


int main(int argc, char *argv[]) {
    NTI_INIT_TIME();
    printf("=========================================\n");
    printf("Matrix A size: %dx%dx%d\n", B, M, K);
    printf("Matrix B size: %dx%dx%d\n", B, N, K);
    printf("Matrix C size: %dx%dx%d\n", B, M, N);
    printf("=========================================\n\n");


    /*======================= Prepare =======================*/
    int sizeA = B * M * K;
    int sizeB = B * N * K;
    int sizeC = B * M * N;
    float *matA = (float*)malloc(sizeof(float) * sizeA);
    float *matB = (float*)malloc(sizeof(float) * sizeB);
    if (matA == NULL || matB == NULL) {
        printf("matA or matB malloc failed!!\n");
    }

    float *matC00 = (float*)malloc(sizeof(float) * sizeC);
    float *matC01 = (float*)malloc(sizeof(float) * sizeC);
    float *matC02 = (float*)malloc(sizeof(float) * sizeC);
    float *matC03 = (float*)malloc(sizeof(float) * sizeC);
    float *matC04 = (float*)malloc(sizeof(float) * sizeC);
    if (matC00 == NULL || 
        matC01 == NULL ||
        matC02 == NULL ||
        matC03 == NULL ||
        matC04 == NULL) {
        printf("matC malloc failed!!\n\n");
    }

    //随机初始化矩阵A和B
    randn<float>(matA, sizeA);
    randn<float>(matB, sizeB);

    /*======================= Calculate =======================*/
    NTI_START_TIME();
    matmul(matA, matB, matC00);
    NTI_END_TIME("Matmul process ...");
    dataShow(matC00);

    NTI_START_TIME();
    matmulN(matA, matB, matC01);
    NTI_END_TIME("Matmul-N process ...");
    dataShow(matC01);

    NTI_START_TIME();
    matmulMN(matA, matB, matC02);
    NTI_END_TIME("Matmul-MN process ...");
    dataShow(matC02);

    NTI_START_TIME();
    matmulMNK(matA, matB, matC03);
    NTI_END_TIME("Matmul-MNK process ...");
    dataShow(matC03);

    NTI_START_TIME();
    matmulMNKOpt(matA, matB, matC04);
    NTI_END_TIME("Matmul-MNKOpt process ...");
    dataShow(matC04);

    return 0;
}