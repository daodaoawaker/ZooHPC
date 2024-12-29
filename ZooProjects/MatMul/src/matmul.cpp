#include <cstring>
// #include "NEON_2_SSE.h"

#if __ARM_NEON
#include "arm_neon.h"
#endif

#include "utils.h"
#include "matmul.h"


/**
 * Size A: B x M x K
 * Size B: B x N x K
 * Size C: B x M x N
*/
int matmul(float *matA, float *matB, float *matC) 
{
    memset((float *)matC, 0, sizeof(float)*B*M*N);
    for (size_t bi = 0; bi < B; bi++) {
        for (size_t mi = 0; mi < M; mi++) {
            for (size_t ni = 0; ni < N; ni++) {
                float sum = 0.0f;
                for (size_t ki = 0; ki < K; ki++) {
                    sum += matA[bi*M*K + mi*K + ki] * matB[bi*N*K + ni*K + ki];
                }
                matC[bi*M*N + mi*N + ni] = sum;
            }
        }
    }

    return 0;
}


/**
 * Size A: B x M x K
 * Size B: B x N x K
 * Size C: B x M x N
 * 
 * 对于输出矩阵，仅在N维度上进行循环展开以改进访存局部性
*/
int matmulN(float *matA, float *matB, float *matC) {
    for (size_t bi = 0; bi < B; bi++) {
        auto pBatchA = matA + bi * M * K;
        auto pBatchB = matB + bi * N * K;
        auto pBatchC = matC + bi * M * N;

        for (size_t mi = 0; mi < M; mi++) {
            auto pRowC = pBatchC + mi * N;
            auto rowA = pBatchA + mi * K;

            /* 在N维度上进行loop unrolling */
            size_t ni = 0;
            for (; ni < N - 3; ni += 4) {
                float r0c0C = 0.0f;
                float r0c1C = 0.0f;
                float r0c2C = 0.0f;
                float r0c3C = 0.0f;
                auto pR0B = pBatchB + (ni + 0) * K;
                auto pR1B = pBatchB + (ni + 1) * K;
                auto pR2B = pBatchB + (ni + 2) * K;
                auto pR3B = pBatchB + (ni + 3) * K;
                // 利用寄存器存储数据以减少对矩阵A的访存操作数
                for (size_t ki = 0; ki < K; ki++) {
                    float dataA = rowA[ki];
                    r0c0C += dataA * pR0B[ki];
                    r0c1C += dataA * pR1B[ki];
                    r0c2C += dataA * pR2B[ki];
                    r0c3C += dataA * pR3B[ki];
                }
                pRowC[ni + 0] = r0c0C;
                pRowC[ni + 1] = r0c1C;
                pRowC[ni + 2] = r0c2C;
                pRowC[ni + 3] = r0c3C;
            }

            /* 处理N维度上的尾循环 */
            for (; ni < N; ni++) {
                float sum = 0.0f;
                auto rowB = pBatchB + ni * K;
                for (size_t ki = 0; ki < K; ki++) {
                    sum += rowA[ki] * rowB[ki];
                }
                pRowC[ni] = sum;
            }
        }
    }

    return 0;
}


/**
 * Size A: B x M x K
 * Size B: B x N x K
 * Size C: B x M x N
 * 
 * 对于输出矩阵，在M/N两个维度上进行循环展开以改进访存局部性
*/
int matmulMN(float *matA, float *matB, float *matC) {
    for (size_t bi = 0; bi < B; bi++) {
        auto pBatchA = matA + bi * M * K;
        auto pBatchB = matB + bi * N * K;
        auto pBatchC = matC + bi * M * N;

        /* 在M维度上进行loop unrolling */
        size_t mi = 0;
        for (; mi < M - 3; mi += 4) {
            auto pR0C = pBatchC + (mi + 0) * N;
            auto pR1C = pBatchC + (mi + 1) * N;
            auto pR2C = pBatchC + (mi + 2) * N;
            auto pR3C = pBatchC + (mi + 3) * N;
            auto pR0A = pBatchA + (mi + 0) * K;
            auto pR1A = pBatchA + (mi + 1) * K;
            auto pR2A = pBatchA + (mi + 2) * K;
            auto pR3A = pBatchA + (mi + 3) * K;

            /* 在N维度上进行loop unrolling */
            size_t ni = 0;
            for (; ni < N - 3; ni += 4) {
                float r0c0C = 0.0f;
                float r0c1C = 0.0f;
                float r0c2C = 0.0f;
                float r0c3C = 0.0f;
                float r1c0C = 0.0f;
                float r1c1C = 0.0f;
                float r1c2C = 0.0f;
                float r1c3C = 0.0f;
                float r2c0C = 0.0f;
                float r2c1C = 0.0f;
                float r2c2C = 0.0f;
                float r2c3C = 0.0f;
                float r3c0C = 0.0f;
                float r3c1C = 0.0f;
                float r3c2C = 0.0f;
                float r3c3C = 0.0f;
                auto pR0B = pBatchB + (ni + 0) * K;
                auto pR1B = pBatchB + (ni + 1) * K;
                auto pR2B = pBatchB + (ni + 2) * K;
                auto pR3B = pBatchB + (ni + 3) * K;
                // 利用寄存器存储数据以实现矩阵A和B中的数据复用
                for (size_t ki = 0; ki < K; ki++) {
                    float r0ckA = pR0A[ki];
                    float r1ckA = pR1A[ki];
                    float r2ckA = pR2A[ki];
                    float r3ckA = pR3A[ki];
                    float r0ckB = pR0B[ki];
                    float r1ckB = pR1B[ki];
                    float r2ckB = pR2B[ki];
                    float r3ckB = pR3B[ki];
                    r0c0C += r0ckA * r0ckB;
                    r0c1C += r0ckA * r1ckB;
                    r0c2C += r0ckA * r2ckB;
                    r0c3C += r0ckA * r3ckB;
                    r1c0C += r1ckA * r0ckB;
                    r1c1C += r1ckA * r1ckB;
                    r1c2C += r1ckA * r2ckB;
                    r1c3C += r1ckA * r3ckB;
                    r2c0C += r2ckA * r0ckB;
                    r2c1C += r2ckA * r1ckB;
                    r2c2C += r2ckA * r2ckB;
                    r2c3C += r2ckA * r3ckB;
                    r3c0C += r3ckA * r0ckB;
                    r3c1C += r3ckA * r1ckB;
                    r3c2C += r3ckA * r2ckB;
                    r3c3C += r3ckA * r3ckB;
                }
                pR0C[ni + 0] = r0c0C;
                pR0C[ni + 1] = r0c1C;
                pR0C[ni + 2] = r0c2C;
                pR0C[ni + 3] = r0c3C;
                pR1C[ni + 0] = r1c0C;
                pR1C[ni + 1] = r1c1C;
                pR1C[ni + 2] = r1c2C;
                pR1C[ni + 3] = r1c3C;
                pR2C[ni + 0] = r2c0C;
                pR2C[ni + 1] = r2c1C;
                pR2C[ni + 2] = r2c2C;
                pR2C[ni + 3] = r2c3C;
                pR3C[ni + 0] = r3c0C;
                pR3C[ni + 1] = r3c1C;
                pR3C[ni + 2] = r3c2C;
                pR3C[ni + 3] = r3c3C;
            }

            /* 处理N维度上的尾循环 */
            for (; ni < N; ni++) {
                float r0C = 0.0f;
                float r1C = 0.0f;
                float r2C = 0.0f;
                float r3C = 0.0f;
                auto pRowB = pBatchB + ni * K;
                for (size_t ki = 0; ki < K; ki++) {
                    float dataB = pRowB[ki];
                    r0C += pR0A[ki] * dataB;
                    r1C += pR1A[ki] * dataB;
                    r2C += pR2A[ki] * dataB;
                    r3C += pR3A[ki] * dataB;
                }
                pR0C[ni] = r0C;
                pR1C[ni] = r1C;
                pR2C[ni] = r2C;
                pR3C[ni] = r3C;
            }
        }

        /* 处理M维度上的尾循环 */
        for (; mi < M; mi++) {
            auto pRowC = pBatchC + mi * N;
            auto pRowA = pBatchA + mi * K;

            /* 在N维度上进行loop unrolling */
            size_t ni = 0;
            for (; ni < N - 3; ni += 4) {
                float c0C = 0.0f;
                float c1C = 0.0f;
                float c2C = 0.0f;
                float c3C = 0.0f;
                auto pR0B = pBatchB + (ni + 0) * K;
                auto pR1B = pBatchB + (ni + 1) * K;
                auto pR2B = pBatchB + (ni + 2) * K;
                auto pR3B = pBatchB + (ni + 3) * K;
                for (size_t ki = 0; ki < K; ki++) {
                    float dataA = pRowA[ki];
                    c0C += dataA * pR0B[ki];
                    c1C += dataA * pR1B[ki];
                    c2C += dataA * pR2B[ki];
                    c3C += dataA * pR3B[ki];
                }
                pRowC[ni + 0] = c0C;
                pRowC[ni + 1] = c1C;
                pRowC[ni + 2] = c2C;
                pRowC[ni + 3] = c3C;
            }

            /* 处理N维度上的尾循环 */
            for (; ni < N; ni++) {
                float sum = 0.0f;
                auto pRowB = pBatchB + ni * K;
                for (size_t ki = 0; ki < K; ki++) {
                    sum += pRowA[ki] * pRowB[ki];
                }
                pRowC[ni] = sum;
            }
        }
    }

    return 0;
}


/**
 * Size A: B x M x K
 * Size B: B x N x K
 * Size C: B x M x N
 * 
 * 对于输出矩阵，在M/N/K三个维度上进行循环展开以改进访存局部性，
 * 同时使用NEON指令进行向量化加速。
*/
int matmulMNK(float *matA, float *matB, float *matC) {
    for (size_t bi = 0; bi < B; bi++) {
        auto pBatchA = matA + bi * M * K;
        auto pBatchB = matB + bi * N * K;
        auto pBatchC = matC + bi * M * N;

        /* 在M维度上进行loop unrolling */
        size_t mi = 0;
        for (; mi < M - 3; mi += 4) {
            auto pR0C = pBatchC + (mi + 0) * N;
            auto pR1C = pBatchC + (mi + 1) * N;
            auto pR2C = pBatchC + (mi + 2) * N;
            auto pR3C = pBatchC + (mi + 3) * N;
            auto pR0A = pBatchA + (mi + 0) * K;
            auto pR1A = pBatchA + (mi + 1) * K;
            auto pR2A = pBatchA + (mi + 2) * K;
            auto pR3A = pBatchA + (mi + 3) * K;

            /* 在N维度上进行loop unrolling */
            size_t ni = 0;
            for (; ni < N - 3; ni += 4) {
                auto pR0B = pBatchB + (ni + 0) * K;
                auto pR1B = pBatchB + (ni + 1) * K;
                auto pR2B = pBatchB + (ni + 2) * K;
                auto pR3B = pBatchB + (ni + 3) * K;
#if __ARM_NEON
                float32x4_t vecR0C = vdupq_n_f32(0.0f);
                float32x4_t vecR1C = vdupq_n_f32(0.0f);
                float32x4_t vecR2C = vdupq_n_f32(0.0f);
                float32x4_t vecR3C = vdupq_n_f32(0.0f);
#else
                float r0c0C = 0.0f;
                float r0c1C = 0.0f;
                float r0c2C = 0.0f;
                float r0c3C = 0.0f;
                float r1c0C = 0.0f;
                float r1c1C = 0.0f;
                float r1c2C = 0.0f;
                float r1c3C = 0.0f;
                float r2c0C = 0.0f;
                float r2c1C = 0.0f;
                float r2c2C = 0.0f;
                float r2c3C = 0.0f;
                float r3c0C = 0.0f;
                float r3c1C = 0.0f;
                float r3c2C = 0.0f;
                float r3c3C = 0.0f;
#endif

                /* 在K维度上进行loop unrolling */
                size_t ki = 0;
                for (; ki < K - 3; ki += 4) {
#if __ARM_NEON
                    float32x4_t vecR0A = vld1q_f32(pR0A + ki);
                    float32x4_t vecR1A = vld1q_f32(pR1A + ki);
                    float32x4_t vecR2A = vld1q_f32(pR2A + ki);
                    float32x4_t vecR3A = vld1q_f32(pR3A + ki);
                    float32x4_t vecR0B = vld1q_f32(pR0B + ki);
                    float32x4_t vecR1B = vld1q_f32(pR1B + ki);
                    float32x4_t vecR2B = vld1q_f32(pR2B + ki);
                    float32x4_t vecR3B = vld1q_f32(pR3B + ki);
                    float32x4_t vecC0B = {
                        vecR0B[0], vecR1B[0], vecR2B[0], vecR3B[0]
                    };
                    float32x4_t vecC1B = {
                        vecR0B[1], vecR1B[1], vecR2B[1], vecR3B[1]
                    };
                    float32x4_t vecC2B = {
                        vecR0B[2], vecR1B[2], vecR2B[2], vecR3B[2]
                    };
                    float32x4_t vecC3B = {
                        vecR0B[3], vecR1B[3], vecR2B[3], vecR3B[3]
                    };

                    vecR0C = vmlaq_laneq_f32(vecR0C, vecC0B, vecR0A, 0);
                    vecR0C = vmlaq_laneq_f32(vecR0C, vecC1B, vecR0A, 1);
                    vecR0C = vmlaq_laneq_f32(vecR0C, vecC2B, vecR0A, 2);
                    vecR0C = vmlaq_laneq_f32(vecR0C, vecC3B, vecR0A, 3);
                    vecR1C = vmlaq_laneq_f32(vecR1C, vecC0B, vecR1A, 0);
                    vecR1C = vmlaq_laneq_f32(vecR1C, vecC1B, vecR1A, 1);
                    vecR1C = vmlaq_laneq_f32(vecR1C, vecC2B, vecR1A, 2);
                    vecR1C = vmlaq_laneq_f32(vecR1C, vecC3B, vecR1A, 3);
                    vecR2C = vmlaq_laneq_f32(vecR2C, vecC0B, vecR2A, 0);
                    vecR2C = vmlaq_laneq_f32(vecR2C, vecC1B, vecR2A, 1);
                    vecR2C = vmlaq_laneq_f32(vecR2C, vecC2B, vecR2A, 2);
                    vecR2C = vmlaq_laneq_f32(vecR2C, vecC3B, vecR2A, 3);
                    vecR3C = vmlaq_laneq_f32(vecR3C, vecC0B, vecR3A, 0);
                    vecR3C = vmlaq_laneq_f32(vecR3C, vecC1B, vecR3A, 1);
                    vecR3C = vmlaq_laneq_f32(vecR3C, vecC2B, vecR3A, 2);
                    vecR3C = vmlaq_laneq_f32(vecR3C, vecC3B, vecR3A, 3);

#else
                    float r0c0A = pR0A[ki + 0];
                    float r0c1A = pR0A[ki + 1];
                    float r0c2A = pR0A[ki + 2];
                    float r0c3A = pR0A[ki + 3];
                    float r1c0A = pR1A[ki + 0];
                    float r1c1A = pR1A[ki + 1];
                    float r1c2A = pR1A[ki + 2];
                    float r1c3A = pR1A[ki + 3];
                    float r2c0A = pR2A[ki + 0];
                    float r2c1A = pR2A[ki + 1];
                    float r2c2A = pR2A[ki + 2];
                    float r2c3A = pR2A[ki + 3];
                    float r3c0A = pR3A[ki + 0];
                    float r3c1A = pR3A[ki + 1];
                    float r3c2A = pR3A[ki + 2];
                    float r3c3A = pR3A[ki + 3];
                    float r0c0B = pR0B[ki + 0];
                    float r0c1B = pR0B[ki + 1];
                    float r0c2B = pR0B[ki + 2];
                    float r0c3B = pR0B[ki + 3];
                    float r1c0B = pR1B[ki + 0];
                    float r1c1B = pR1B[ki + 1];
                    float r1c2B = pR1B[ki + 2];
                    float r1c3B = pR1B[ki + 3];
                    float r2c0B = pR2B[ki + 0];
                    float r2c1B = pR2B[ki + 1];
                    float r2c2B = pR2B[ki + 2];
                    float r2c3B = pR2B[ki + 3];
                    float r3c0B = pR3B[ki + 0];
                    float r3c1B = pR3B[ki + 1];
                    float r3c2B = pR3B[ki + 2];
                    float r3c3B = pR3B[ki + 3];

                    r0c0C += r0c0A * r0c0B;
                    r0c0C += r0c1A * r0c1B;
                    r0c0C += r0c2A * r0c2B;
                    r0c0C += r0c3A * r0c3B;
                    r0c1C += r0c0A * r1c0B;
                    r0c1C += r0c1A * r1c1B;
                    r0c1C += r0c2A * r1c2B;
                    r0c1C += r0c3A * r1c3B;
                    r0c2C += r0c0A * r2c0B;
                    r0c2C += r0c1A * r2c1B;
                    r0c2C += r0c2A * r2c2B;
                    r0c2C += r0c3A * r2c3B;
                    r0c3C += r0c0A * r3c0B;
                    r0c3C += r0c1A * r3c1B;
                    r0c3C += r0c2A * r3c2B;
                    r0c3C += r0c3A * r3c3B;

                    r1c0C += r1c0A * r0c0B;
                    r1c0C += r1c1A * r0c1B;
                    r1c0C += r1c2A * r0c2B;
                    r1c0C += r1c3A * r0c3B;
                    r1c1C += r1c0A * r1c0B;
                    r1c1C += r1c1A * r1c1B;
                    r1c1C += r1c2A * r1c2B;
                    r1c1C += r1c3A * r1c3B;
                    r1c2C += r1c0A * r2c0B;
                    r1c2C += r1c1A * r2c1B;
                    r1c2C += r1c2A * r2c2B;
                    r1c2C += r1c3A * r2c3B;
                    r1c3C += r1c0A * r3c0B;
                    r1c3C += r1c1A * r3c1B;
                    r1c3C += r1c2A * r3c2B;
                    r1c3C += r1c3A * r3c3B;

                    r2c0C += r2c0A * r0c0B;
                    r2c0C += r2c1A * r0c1B;
                    r2c0C += r2c2A * r0c2B;
                    r2c0C += r2c3A * r0c3B;
                    r2c1C += r2c0A * r1c0B;
                    r2c1C += r2c1A * r1c1B;
                    r2c1C += r2c2A * r1c2B;
                    r2c1C += r2c3A * r1c3B;
                    r2c2C += r2c0A * r2c0B;
                    r2c2C += r2c1A * r2c1B;
                    r2c2C += r2c2A * r2c2B;
                    r2c2C += r2c3A * r2c3B;
                    r2c3C += r2c0A * r3c0B;
                    r2c3C += r2c1A * r3c1B;
                    r2c3C += r2c2A * r3c2B;
                    r2c3C += r2c3A * r3c3B;

                    r3c0C += r3c0A * r0c0B;
                    r3c0C += r3c1A * r0c1B;
                    r3c0C += r3c2A * r0c2B;
                    r3c0C += r3c3A * r0c3B;
                    r3c1C += r3c0A * r1c0B;
                    r3c1C += r3c1A * r1c1B;
                    r3c1C += r3c2A * r1c2B;
                    r3c1C += r3c3A * r1c3B;
                    r3c2C += r3c0A * r2c0B;
                    r3c2C += r3c1A * r2c1B;
                    r3c2C += r3c2A * r2c2B;
                    r3c2C += r3c3A * r2c3B;
                    r3c3C += r3c0A * r3c0B;
                    r3c3C += r3c1A * r3c1B;
                    r3c3C += r3c2A * r3c2B;
                    r3c3C += r3c3A * r3c3B;
#endif
                }

                /* 处理K维度上的尾循环 */
                for (; ki < K; ki++) {
                    float r0ckA = pR0A[ki];
                    float r1ckA = pR1A[ki];
                    float r2ckA = pR2A[ki];
                    float r3ckA = pR3A[ki];
                    float r0ckB = pR0B[ki];
                    float r1ckB = pR1B[ki];
                    float r2ckB = pR2B[ki];
                    float r3ckB = pR3B[ki];
#if __ARM_NEON
                    float32x4_t vecCkB = {
                        r0ckB, r1ckB, r2ckB, r3ckB
                    };
                    vecR0C = vmlaq_n_f32(vecR0C, vecCkB, r0ckA);
                    vecR1C = vmlaq_n_f32(vecR1C, vecCkB, r1ckA);
                    vecR2C = vmlaq_n_f32(vecR2C, vecCkB, r2ckA);
                    vecR3C = vmlaq_n_f32(vecR3C, vecCkB, r3ckA);
#else
                    r0c0C += r0ckA * r0ckB;
                    r0c1C += r0ckA * r1ckB;
                    r0c2C += r0ckA * r2ckB;
                    r0c3C += r0ckA * r3ckB;
                    r1c0C += r1ckA * r0ckB;
                    r1c1C += r1ckA * r1ckB;
                    r1c2C += r1ckA * r2ckB;
                    r1c3C += r1ckA * r3ckB;
                    r2c0C += r2ckA * r0ckB;
                    r2c1C += r2ckA * r1ckB;
                    r2c2C += r2ckA * r2ckB;
                    r2c3C += r2ckA * r3ckB;
                    r3c0C += r3ckA * r0ckB;
                    r3c1C += r3ckA * r1ckB;
                    r3c2C += r3ckA * r2ckB;
                    r3c3C += r3ckA * r3ckB;
#endif
                }

#if __ARM_NEON
                vst1q_f32(pR0C + ni, vecR0C);
                vst1q_f32(pR1C + ni, vecR1C);
                vst1q_f32(pR2C + ni, vecR2C);
                vst1q_f32(pR3C + ni, vecR3C);
#else
                pR0C[ni + 0] = r0c0C;
                pR0C[ni + 1] = r0c1C;
                pR0C[ni + 2] = r0c2C;
                pR0C[ni + 3] = r0c3C;
                pR1C[ni + 0] = r1c0C;
                pR1C[ni + 1] = r1c1C;
                pR1C[ni + 2] = r1c2C;
                pR1C[ni + 3] = r1c3C;
                pR2C[ni + 0] = r2c0C;
                pR2C[ni + 1] = r2c1C;
                pR2C[ni + 2] = r2c2C;
                pR2C[ni + 3] = r2c3C;
                pR3C[ni + 0] = r3c0C;
                pR3C[ni + 1] = r3c1C;
                pR3C[ni + 2] = r3c2C;
                pR3C[ni + 3] = r3c3C;
#endif
            }

            /* 处理N维度上的尾循环 */
            for (; ni < N; ni++) {
                float r0C = 0.0f;
                float r1C = 0.0f;
                float r2C = 0.0f;
                float r3C = 0.0f;
                auto pRowB = pBatchB + ni * K;

                /* 在K维度上进行loop unrolling */
                size_t ki = 0;
                for (; ki < K - 3; ki += 4) {
                    float r0c0A = pR0A[ki + 0];
                    float r0c1A = pR0A[ki + 1];
                    float r0c2A = pR0A[ki + 2];
                    float r0c3A = pR0A[ki + 3];
                    float r1c0A = pR1A[ki + 0];
                    float r1c1A = pR1A[ki + 1];
                    float r1c2A = pR1A[ki + 2];
                    float r1c3A = pR1A[ki + 3];
                    float r2c0A = pR2A[ki + 0];
                    float r2c1A = pR2A[ki + 1];
                    float r2c2A = pR2A[ki + 2];
                    float r2c3A = pR2A[ki + 3];
                    float r3c0A = pR3A[ki + 0];
                    float r3c1A = pR3A[ki + 1];
                    float r3c2A = pR3A[ki + 2];
                    float r3c3A = pR3A[ki + 3];
                    float rnc0B = pRowB[ki + 0];
                    float rnc1B = pRowB[ki + 1];
                    float rnc2B = pRowB[ki + 2];
                    float rnc3B = pRowB[ki + 3];
                    r0C += r0c0A * rnc0B;
                    r0C += r0c1A * rnc1B;
                    r0C += r0c2A * rnc2B;
                    r0C += r0c3A * rnc3B;
                    r1C += r1c0A * rnc0B;
                    r1C += r1c1A * rnc1B;
                    r1C += r1c2A * rnc2B;
                    r1C += r1c3A * rnc3B;
                    r2C += r2c0A * rnc0B;
                    r2C += r2c1A * rnc1B;
                    r2C += r2c2A * rnc2B;
                    r2C += r2c3A * rnc3B;
                    r3C += r3c0A * rnc0B;
                    r3C += r3c1A * rnc1B;
                    r3C += r3c2A * rnc2B;
                    r3C += r3c3A * rnc3B;
                }

                /* 处理K维度上的尾循环 */
                for (; ki < K; ki++) {
                    float dataB = pRowB[ki];
                    r0C += pR0A[ki] * dataB;
                    r1C += pR1A[ki] * dataB;
                    r2C += pR2A[ki] * dataB;
                    r3C += pR3A[ki] * dataB;
                }

                pR0C[ni] = r0C;
                pR1C[ni] = r1C;
                pR2C[ni] = r2C;
                pR3C[ni] = r3C;
            }
        }

        /* 处理M维度上的尾循环 */
        for (; mi < M; mi++) {
            auto pRowC = pBatchC + mi * N;
            auto pRowA = pBatchA + mi * K;

            /* 在N维度上进行loop unrolling */
            size_t ni = 0;
            for (; ni < N - 3; ni += 4) {
                auto pR0B = pBatchB + (ni + 0) * K;
                auto pR1B = pBatchB + (ni + 1) * K;
                auto pR2B = pBatchB + (ni + 2) * K;
                auto pR3B = pBatchB + (ni + 3) * K;
#if __ARM_NEON
                float32x4_t vecRowC = vdupq_n_f32(0.0f);
#else
                float c0C = 0.0f;
                float c1C = 0.0f;
                float c2C = 0.0f;
                float c3C = 0.0f;
#endif

                /* 在K维度上进行loop unrolling */
                size_t ki = 0;
                for (; ki < K - 3; ki += 4) {
                    float rmc0A = pRowA[ki + 0];
                    float rmc1A = pRowA[ki + 1];
                    float rmc2A = pRowA[ki + 2];
                    float rmc3A = pRowA[ki + 3];
                    float r0c0B = pR0B[ki + 0];
                    float r0c1B = pR0B[ki + 1];
                    float r0c2B = pR0B[ki + 2];
                    float r0c3B = pR0B[ki + 3];
                    float r1c0B = pR1B[ki + 0];
                    float r1c1B = pR1B[ki + 1];
                    float r1c2B = pR1B[ki + 2];
                    float r1c3B = pR1B[ki + 3];
                    float r2c0B = pR2B[ki + 0];
                    float r2c1B = pR2B[ki + 1];
                    float r2c2B = pR2B[ki + 2];
                    float r2c3B = pR2B[ki + 3];
                    float r3c0B = pR3B[ki + 0];
                    float r3c1B = pR3B[ki + 1];
                    float r3c2B = pR3B[ki + 2];
                    float r3c3B = pR3B[ki + 3];
#if __ARM_NEON
                    float32x4_t vecC0B = {r0c0B, r1c0B, r2c0B, r3c0B};
                    float32x4_t vecC1B = {r0c1B, r1c1B, r2c1B, r3c1B};
                    float32x4_t vecC2B = {r0c2B, r1c2B, r2c2B, r3c2B};
                    float32x4_t vecC3B = {r0c3B, r1c3B, r2c3B, r3c3B};
                    vecRowC = vmlaq_n_f32(vecRowC, vecC0B, rmc0A);
                    vecRowC = vmlaq_n_f32(vecRowC, vecC1B, rmc1A);
                    vecRowC = vmlaq_n_f32(vecRowC, vecC2B, rmc2A);
                    vecRowC = vmlaq_n_f32(vecRowC, vecC3B, rmc3A);
#else
                    c0C += rmc0A * r0c0B;
                    c0C += rmc1A * r0c1B;
                    c0C += rmc2A * r0c2B;
                    c0C += rmc3A * r0c3B;
                    c1C += rmc0A * r1c0B;
                    c1C += rmc1A * r1c1B;
                    c1C += rmc2A * r1c2B;
                    c1C += rmc3A * r1c3B;
                    c2C += rmc0A * r2c0B;
                    c2C += rmc1A * r2c1B;
                    c2C += rmc2A * r2c2B;
                    c2C += rmc3A * r2c3B;
                    c3C += rmc0A * r3c0B;
                    c3C += rmc1A * r3c1B;
                    c3C += rmc2A * r3c2B;
                    c3C += rmc3A * r3c3B;
#endif
                }

                /* 处理K维度上的尾循环 */
                for (; ki < K; ki++) {
                    float dataA = pRowA[ki];
                    float r0ckB = pR0B[ki];
                    float r1ckB = pR1B[ki];
                    float r2ckB = pR2B[ki];
                    float r3ckB = pR3B[ki];
#if __ARM_NEON
                    float32x4_t vecColB = {r0ckB, r1ckB, r2ckB, r3ckB};
                    vecRowC = vmlaq_n_f32(vecRowC, vecColB, dataA);
#else
                    c0C += dataA * r0ckB;
                    c1C += dataA * r1ckB;
                    c2C += dataA * r2ckB;
                    c3C += dataA * r3ckB;
#endif
                }

#if __ARM_NEON
                vst1q_f32(pRowC + ni, vecRowC);
#else
                pRowC[ni + 0] = c0C;
                pRowC[ni + 1] = c1C;
                pRowC[ni + 2] = c2C;
                pRowC[ni + 3] = c3C;
#endif
            }

            /* 处理N维度上的尾循环 */
            for (; ni < N; ni++) {
                float sum = 0.0f;
                auto pRowB = pBatchB + ni * K;
                for (int ki = 0; ki < K; ki++) {
                    sum += pRowA[ki] * pRowB[ki];
                }
                pRowC[ni] = sum;
            }
        }
    }

    return 0;
}


/**
 * Size A: B x M x K
 * Size B: B x N x K
 * Size C: B x M x N
 * 
 * 对于输出矩阵，在M/N/K三个维度上进行循环展开以改进访存局部性，使用更激进的分块计算，
 * 同时使用NEON指令进行向量化加速。
*/
int matmulMNKOpt(float *matA, float *matB, float *matC) {
    for (size_t bi = 0; bi < B; bi++) {
        auto pBatchA = matA + bi * M * K;
        auto pBatchB = matB + bi * N * K;
        auto pBatchC = matC + bi * M * N;

        /* 在M维度上进行loop unrolling */
        size_t mi = 0;
        for (; mi < M - 7; mi += 8) {
            auto pR0C = pBatchC + (mi + 0) * N;
            auto pR1C = pBatchC + (mi + 1) * N;
            auto pR2C = pBatchC + (mi + 2) * N;
            auto pR3C = pBatchC + (mi + 3) * N;
            auto pR4C = pBatchC + (mi + 4) * N;
            auto pR5C = pBatchC + (mi + 5) * N;
            auto pR6C = pBatchC + (mi + 6) * N;
            auto pR7C = pBatchC + (mi + 7) * N;
            auto pR0A = pBatchA + (mi + 0) * K;
            auto pR1A = pBatchA + (mi + 1) * K;
            auto pR2A = pBatchA + (mi + 2) * K;
            auto pR3A = pBatchA + (mi + 3) * K;
            auto pR4A = pBatchA + (mi + 4) * K;
            auto pR5A = pBatchA + (mi + 5) * K;
            auto pR6A = pBatchA + (mi + 6) * K;
            auto pR7A = pBatchA + (mi + 7) * K;

            /* 在N维度上进行loop unrolling */
            size_t ni = 0;
            for (; ni < N - 7; ni += 8) {
                auto pR0B = pBatchB + (ni + 0) * K;
                auto pR1B = pBatchB + (ni + 1) * K;
                auto pR2B = pBatchB + (ni + 2) * K;
                auto pR3B = pBatchB + (ni + 3) * K;
                auto pR4B = pBatchB + (ni + 4) * K;
                auto pR5B = pBatchB + (ni + 5) * K;
                auto pR6B = pBatchB + (ni + 6) * K;
                auto pR7B = pBatchB + (ni + 7) * K;
#if __ARM_NEON
                float32x4_t vecR0P0C = vdupq_n_f32(0.0f);
                float32x4_t vecR0P1C = vdupq_n_f32(0.0f);
                float32x4_t vecR1P0C = vdupq_n_f32(0.0f);
                float32x4_t vecR1P1C = vdupq_n_f32(0.0f);
                float32x4_t vecR2P0C = vdupq_n_f32(0.0f);
                float32x4_t vecR2P1C = vdupq_n_f32(0.0f);
                float32x4_t vecR3P0C = vdupq_n_f32(0.0f);
                float32x4_t vecR3P1C = vdupq_n_f32(0.0f);
                float32x4_t vecR4P0C = vdupq_n_f32(0.0f);
                float32x4_t vecR4P1C = vdupq_n_f32(0.0f);
                float32x4_t vecR5P0C = vdupq_n_f32(0.0f);
                float32x4_t vecR5P1C = vdupq_n_f32(0.0f);
                float32x4_t vecR6P0C = vdupq_n_f32(0.0f);
                float32x4_t vecR6P1C = vdupq_n_f32(0.0f);
                float32x4_t vecR7P0C = vdupq_n_f32(0.0f);
                float32x4_t vecR7P1C = vdupq_n_f32(0.0f);
#else
                float r0c0C = 0.0f;
                float r0c1C = 0.0f;
                float r0c2C = 0.0f;
                float r0c3C = 0.0f;
                float r0c4C = 0.0f;
                float r0c5C = 0.0f;
                float r0c6C = 0.0f;
                float r0c7C = 0.0f;
                float r1c0C = 0.0f;
                float r1c1C = 0.0f;
                float r1c2C = 0.0f;
                float r1c3C = 0.0f;
                float r1c4C = 0.0f;
                float r1c5C = 0.0f;
                float r1c6C = 0.0f;
                float r1c7C = 0.0f;
                float r2c0C = 0.0f;
                float r2c1C = 0.0f;
                float r2c2C = 0.0f;
                float r2c3C = 0.0f;
                float r2c4C = 0.0f;
                float r2c5C = 0.0f;
                float r2c6C = 0.0f;
                float r2c7C = 0.0f;
                float r3c0C = 0.0f;
                float r3c1C = 0.0f;
                float r3c2C = 0.0f;
                float r3c3C = 0.0f;
                float r3c4C = 0.0f;
                float r3c5C = 0.0f;
                float r3c6C = 0.0f;
                float r3c7C = 0.0f;
                float r4c0C = 0.0f;
                float r4c1C = 0.0f;
                float r4c2C = 0.0f;
                float r4c3C = 0.0f;
                float r4c4C = 0.0f;
                float r4c5C = 0.0f;
                float r4c6C = 0.0f;
                float r4c7C = 0.0f;
                float r5c0C = 0.0f;
                float r5c1C = 0.0f;
                float r5c2C = 0.0f;
                float r5c3C = 0.0f;
                float r5c4C = 0.0f;
                float r5c5C = 0.0f;
                float r5c6C = 0.0f;
                float r5c7C = 0.0f;
                float r6c0C = 0.0f;
                float r6c1C = 0.0f;
                float r6c2C = 0.0f;
                float r6c3C = 0.0f;
                float r6c4C = 0.0f;
                float r6c5C = 0.0f;
                float r6c6C = 0.0f;
                float r6c7C = 0.0f;
                float r7c0C = 0.0f;
                float r7c1C = 0.0f;
                float r7c2C = 0.0f;
                float r7c3C = 0.0f;
                float r7c4C = 0.0f;
                float r7c5C = 0.0f;
                float r7c6C = 0.0f;
                float r7c7C = 0.0f;
#endif

                /* 在K维度上进行loop unrolling */
                size_t ki = 0;
                for (; ki < K - 7; ki += 8) {
#if __ARM_NEON
                    float32x4_t vecR0P0A = vld1q_f32(pR0A + ki);
                    float32x4_t vecR0P1A = vld1q_f32(pR0A + ki + 4);
                    float32x4_t vecR1P0A = vld1q_f32(pR1A + ki);
                    float32x4_t vecR1P1A = vld1q_f32(pR1A + ki + 4);
                    float32x4_t vecR2P0A = vld1q_f32(pR2A + ki);
                    float32x4_t vecR2P1A = vld1q_f32(pR2A + ki + 4);
                    float32x4_t vecR3P0A = vld1q_f32(pR3A + ki);
                    float32x4_t vecR3P1A = vld1q_f32(pR3A + ki + 4);
                    float32x4_t vecR4P0A = vld1q_f32(pR4A + ki);
                    float32x4_t vecR4P1A = vld1q_f32(pR4A + ki + 4);
                    float32x4_t vecR5P0A = vld1q_f32(pR5A + ki);
                    float32x4_t vecR5P1A = vld1q_f32(pR5A + ki + 4);
                    float32x4_t vecR6P0A = vld1q_f32(pR6A + ki);
                    float32x4_t vecR6P1A = vld1q_f32(pR6A + ki + 4);
                    float32x4_t vecR7P0A = vld1q_f32(pR7A + ki);
                    float32x4_t vecR7P1A = vld1q_f32(pR7A + ki + 4);

                    float32x4_t vecR0P0B = vld1q_f32(pR0B + ki);
                    float32x4_t vecR0P1B = vld1q_f32(pR0B + ki + 4);
                    float32x4_t vecR1P0B = vld1q_f32(pR1B + ki);
                    float32x4_t vecR1P1B = vld1q_f32(pR1B + ki + 4);
                    float32x4_t vecR2P0B = vld1q_f32(pR2B + ki);
                    float32x4_t vecR2P1B = vld1q_f32(pR2B + ki + 4);
                    float32x4_t vecR3P0B = vld1q_f32(pR3B + ki);
                    float32x4_t vecR3P1B = vld1q_f32(pR3B + ki + 4);
                    float32x4_t vecR4P0B = vld1q_f32(pR4B + ki);
                    float32x4_t vecR4P1B = vld1q_f32(pR4B + ki + 4);
                    float32x4_t vecR5P0B = vld1q_f32(pR5B + ki);
                    float32x4_t vecR5P1B = vld1q_f32(pR5B + ki + 4);
                    float32x4_t vecR6P0B = vld1q_f32(pR6B + ki);
                    float32x4_t vecR6P1B = vld1q_f32(pR6B + ki + 4);
                    float32x4_t vecR7P0B = vld1q_f32(pR7B + ki);
                    float32x4_t vecR7P1B = vld1q_f32(pR7B + ki + 4);

                    float32x4_t vecC0P0B = {
                        vecR0P0B[0], vecR1P0B[0], vecR2P0B[0], vecR3P0B[0]
                    };
                    float32x4_t vecC0P1B = {
                        vecR4P0B[0], vecR5P0B[0], vecR6P0B[0], vecR7P0B[0]
                    };

                    float32x4_t vecC1P0B = {
                        vecR0P0B[1], vecR1P0B[1], vecR2P0B[1], vecR3P0B[1]
                    };
                    float32x4_t vecC1P1B = {
                        vecR4P0B[1], vecR5P0B[1], vecR6P0B[1], vecR7P0B[1]
                    };

                    float32x4_t vecC2P0B = {
                        vecR0P0B[2], vecR1P0B[2], vecR2P0B[2], vecR3P0B[2]
                    };
                    float32x4_t vecC2P1B = {
                        vecR4P0B[2], vecR5P0B[2], vecR6P0B[2], vecR7P0B[2]
                    };

                    float32x4_t vecC3P0B = {
                        vecR0P0B[3], vecR1P0B[3], vecR2P0B[3], vecR3P0B[3]
                    };
                    float32x4_t vecC3P1B = {
                        vecR4P0B[3], vecR5P0B[3], vecR6P0B[3], vecR7P0B[3]
                    };

                    float32x4_t vecC4P0B = {
                        vecR0P1B[0], vecR1P1B[0], vecR2P1B[0], vecR3P1B[0]
                    };
                    float32x4_t vecC4P1B = {
                        vecR4P1B[0], vecR5P1B[0], vecR6P1B[0], vecR7P1B[0]
                    };

                    float32x4_t vecC5P0B = {
                        vecR0P1B[1], vecR1P1B[1], vecR2P1B[1], vecR3P1B[1]
                    };
                    float32x4_t vecC5P1B = {
                        vecR4P1B[1], vecR5P1B[1], vecR6P1B[1], vecR7P1B[1]
                    };

                    float32x4_t vecC6P0B = {
                        vecR0P1B[2], vecR1P1B[2], vecR2P1B[2], vecR3P1B[2]
                    };
                    float32x4_t vecC6P1B = {
                        vecR4P1B[2], vecR5P1B[2], vecR6P1B[2], vecR7P1B[2]
                    };

                    float32x4_t vecC7P0B = {
                        vecR0P1B[3], vecR1P1B[3], vecR2P1B[3], vecR3P1B[3]
                    };
                    float32x4_t vecC7P1B = {
                        vecR4P1B[3], vecR5P1B[3], vecR6P1B[3], vecR7P1B[3]
                    };

                    vecR0P0C = vmlaq_laneq_f32(vecR0P0C, vecC0P0B, vecR0P0A, 0);
                    vecR0P0C = vmlaq_laneq_f32(vecR0P0C, vecC1P0B, vecR0P0A, 1);
                    vecR0P0C = vmlaq_laneq_f32(vecR0P0C, vecC2P0B, vecR0P0A, 2);
                    vecR0P0C = vmlaq_laneq_f32(vecR0P0C, vecC3P0B, vecR0P0A, 3);
                    vecR0P0C = vmlaq_laneq_f32(vecR0P0C, vecC4P0B, vecR0P1A, 0);
                    vecR0P0C = vmlaq_laneq_f32(vecR0P0C, vecC5P0B, vecR0P1A, 1);
                    vecR0P0C = vmlaq_laneq_f32(vecR0P0C, vecC6P0B, vecR0P1A, 2);
                    vecR0P0C = vmlaq_laneq_f32(vecR0P0C, vecC7P0B, vecR0P1A, 3);
                    vecR0P1C = vmlaq_laneq_f32(vecR0P1C, vecC0P1B, vecR0P0A, 0);
                    vecR0P1C = vmlaq_laneq_f32(vecR0P1C, vecC1P1B, vecR0P0A, 1);
                    vecR0P1C = vmlaq_laneq_f32(vecR0P1C, vecC2P1B, vecR0P0A, 2);
                    vecR0P1C = vmlaq_laneq_f32(vecR0P1C, vecC3P1B, vecR0P0A, 3);
                    vecR0P1C = vmlaq_laneq_f32(vecR0P1C, vecC4P1B, vecR0P1A, 0);
                    vecR0P1C = vmlaq_laneq_f32(vecR0P1C, vecC5P1B, vecR0P1A, 1);
                    vecR0P1C = vmlaq_laneq_f32(vecR0P1C, vecC6P1B, vecR0P1A, 2);
                    vecR0P1C = vmlaq_laneq_f32(vecR0P1C, vecC7P1B, vecR0P1A, 3);

                    vecR1P0C = vmlaq_laneq_f32(vecR1P0C, vecC0P0B, vecR1P0A, 0);
                    vecR1P0C = vmlaq_laneq_f32(vecR1P0C, vecC1P0B, vecR1P0A, 1);
                    vecR1P0C = vmlaq_laneq_f32(vecR1P0C, vecC2P0B, vecR1P0A, 2);
                    vecR1P0C = vmlaq_laneq_f32(vecR1P0C, vecC3P0B, vecR1P0A, 3);
                    vecR1P0C = vmlaq_laneq_f32(vecR1P0C, vecC4P0B, vecR1P1A, 0);
                    vecR1P0C = vmlaq_laneq_f32(vecR1P0C, vecC5P0B, vecR1P1A, 1);
                    vecR1P0C = vmlaq_laneq_f32(vecR1P0C, vecC6P0B, vecR1P1A, 2);
                    vecR1P0C = vmlaq_laneq_f32(vecR1P0C, vecC7P0B, vecR1P1A, 3);
                    vecR1P1C = vmlaq_laneq_f32(vecR1P1C, vecC0P1B, vecR1P0A, 0);
                    vecR1P1C = vmlaq_laneq_f32(vecR1P1C, vecC1P1B, vecR1P0A, 1);
                    vecR1P1C = vmlaq_laneq_f32(vecR1P1C, vecC2P1B, vecR1P0A, 2);
                    vecR1P1C = vmlaq_laneq_f32(vecR1P1C, vecC3P1B, vecR1P0A, 3);
                    vecR1P1C = vmlaq_laneq_f32(vecR1P1C, vecC4P1B, vecR1P1A, 0);
                    vecR1P1C = vmlaq_laneq_f32(vecR1P1C, vecC5P1B, vecR1P1A, 1);
                    vecR1P1C = vmlaq_laneq_f32(vecR1P1C, vecC6P1B, vecR1P1A, 2);
                    vecR1P1C = vmlaq_laneq_f32(vecR1P1C, vecC7P1B, vecR1P1A, 3);

                    vecR2P0C = vmlaq_laneq_f32(vecR2P0C, vecC0P0B, vecR2P0A, 0);
                    vecR2P0C = vmlaq_laneq_f32(vecR2P0C, vecC1P0B, vecR2P0A, 1);
                    vecR2P0C = vmlaq_laneq_f32(vecR2P0C, vecC2P0B, vecR2P0A, 2);
                    vecR2P0C = vmlaq_laneq_f32(vecR2P0C, vecC3P0B, vecR2P0A, 3);
                    vecR2P0C = vmlaq_laneq_f32(vecR2P0C, vecC4P0B, vecR2P1A, 0);
                    vecR2P0C = vmlaq_laneq_f32(vecR2P0C, vecC5P0B, vecR2P1A, 1);
                    vecR2P0C = vmlaq_laneq_f32(vecR2P0C, vecC6P0B, vecR2P1A, 2);
                    vecR2P0C = vmlaq_laneq_f32(vecR2P0C, vecC7P0B, vecR2P1A, 3);
                    vecR2P1C = vmlaq_laneq_f32(vecR2P1C, vecC0P1B, vecR2P0A, 0);
                    vecR2P1C = vmlaq_laneq_f32(vecR2P1C, vecC1P1B, vecR2P0A, 1);
                    vecR2P1C = vmlaq_laneq_f32(vecR2P1C, vecC2P1B, vecR2P0A, 2);
                    vecR2P1C = vmlaq_laneq_f32(vecR2P1C, vecC3P1B, vecR2P0A, 3);
                    vecR2P1C = vmlaq_laneq_f32(vecR2P1C, vecC4P1B, vecR2P1A, 0);
                    vecR2P1C = vmlaq_laneq_f32(vecR2P1C, vecC5P1B, vecR2P1A, 1);
                    vecR2P1C = vmlaq_laneq_f32(vecR2P1C, vecC6P1B, vecR2P1A, 2);
                    vecR2P1C = vmlaq_laneq_f32(vecR2P1C, vecC7P1B, vecR2P1A, 3);

                    vecR3P0C = vmlaq_laneq_f32(vecR3P0C, vecC0P0B, vecR3P0A, 0);
                    vecR3P0C = vmlaq_laneq_f32(vecR3P0C, vecC1P0B, vecR3P0A, 1);
                    vecR3P0C = vmlaq_laneq_f32(vecR3P0C, vecC2P0B, vecR3P0A, 2);
                    vecR3P0C = vmlaq_laneq_f32(vecR3P0C, vecC3P0B, vecR3P0A, 3);
                    vecR3P0C = vmlaq_laneq_f32(vecR3P0C, vecC4P0B, vecR3P1A, 0);
                    vecR3P0C = vmlaq_laneq_f32(vecR3P0C, vecC5P0B, vecR3P1A, 1);
                    vecR3P0C = vmlaq_laneq_f32(vecR3P0C, vecC6P0B, vecR3P1A, 2);
                    vecR3P0C = vmlaq_laneq_f32(vecR3P0C, vecC7P0B, vecR3P1A, 3);
                    vecR3P1C = vmlaq_laneq_f32(vecR3P1C, vecC0P1B, vecR3P0A, 0);
                    vecR3P1C = vmlaq_laneq_f32(vecR3P1C, vecC1P1B, vecR3P0A, 1);
                    vecR3P1C = vmlaq_laneq_f32(vecR3P1C, vecC2P1B, vecR3P0A, 2);
                    vecR3P1C = vmlaq_laneq_f32(vecR3P1C, vecC3P1B, vecR3P0A, 3);
                    vecR3P1C = vmlaq_laneq_f32(vecR3P1C, vecC4P1B, vecR3P1A, 0);
                    vecR3P1C = vmlaq_laneq_f32(vecR3P1C, vecC5P1B, vecR3P1A, 1);
                    vecR3P1C = vmlaq_laneq_f32(vecR3P1C, vecC6P1B, vecR3P1A, 2);
                    vecR3P1C = vmlaq_laneq_f32(vecR3P1C, vecC7P1B, vecR3P1A, 3);

                    vecR4P0C = vmlaq_laneq_f32(vecR4P0C, vecC0P0B, vecR4P0A, 0);
                    vecR4P0C = vmlaq_laneq_f32(vecR4P0C, vecC1P0B, vecR4P0A, 1);
                    vecR4P0C = vmlaq_laneq_f32(vecR4P0C, vecC2P0B, vecR4P0A, 2);
                    vecR4P0C = vmlaq_laneq_f32(vecR4P0C, vecC3P0B, vecR4P0A, 3);
                    vecR4P0C = vmlaq_laneq_f32(vecR4P0C, vecC4P0B, vecR4P1A, 0);
                    vecR4P0C = vmlaq_laneq_f32(vecR4P0C, vecC5P0B, vecR4P1A, 1);
                    vecR4P0C = vmlaq_laneq_f32(vecR4P0C, vecC6P0B, vecR4P1A, 2);
                    vecR4P0C = vmlaq_laneq_f32(vecR4P0C, vecC7P0B, vecR4P1A, 3);
                    vecR4P1C = vmlaq_laneq_f32(vecR4P1C, vecC0P1B, vecR4P0A, 0);
                    vecR4P1C = vmlaq_laneq_f32(vecR4P1C, vecC1P1B, vecR4P0A, 1);
                    vecR4P1C = vmlaq_laneq_f32(vecR4P1C, vecC2P1B, vecR4P0A, 2);
                    vecR4P1C = vmlaq_laneq_f32(vecR4P1C, vecC3P1B, vecR4P0A, 3);
                    vecR4P1C = vmlaq_laneq_f32(vecR4P1C, vecC4P1B, vecR4P1A, 0);
                    vecR4P1C = vmlaq_laneq_f32(vecR4P1C, vecC5P1B, vecR4P1A, 1);
                    vecR4P1C = vmlaq_laneq_f32(vecR4P1C, vecC6P1B, vecR4P1A, 2);
                    vecR4P1C = vmlaq_laneq_f32(vecR4P1C, vecC7P1B, vecR4P1A, 3);

                    vecR5P0C = vmlaq_laneq_f32(vecR5P0C, vecC0P0B, vecR5P0A, 0);
                    vecR5P0C = vmlaq_laneq_f32(vecR5P0C, vecC1P0B, vecR5P0A, 1);
                    vecR5P0C = vmlaq_laneq_f32(vecR5P0C, vecC2P0B, vecR5P0A, 2);
                    vecR5P0C = vmlaq_laneq_f32(vecR5P0C, vecC3P0B, vecR5P0A, 3);
                    vecR5P0C = vmlaq_laneq_f32(vecR5P0C, vecC4P0B, vecR5P1A, 0);
                    vecR5P0C = vmlaq_laneq_f32(vecR5P0C, vecC5P0B, vecR5P1A, 1);
                    vecR5P0C = vmlaq_laneq_f32(vecR5P0C, vecC6P0B, vecR5P1A, 2);
                    vecR5P0C = vmlaq_laneq_f32(vecR5P0C, vecC7P0B, vecR5P1A, 3);
                    vecR5P1C = vmlaq_laneq_f32(vecR5P1C, vecC0P1B, vecR5P0A, 0);
                    vecR5P1C = vmlaq_laneq_f32(vecR5P1C, vecC1P1B, vecR5P0A, 1);
                    vecR5P1C = vmlaq_laneq_f32(vecR5P1C, vecC2P1B, vecR5P0A, 2);
                    vecR5P1C = vmlaq_laneq_f32(vecR5P1C, vecC3P1B, vecR5P0A, 3);
                    vecR5P1C = vmlaq_laneq_f32(vecR5P1C, vecC4P1B, vecR5P1A, 0);
                    vecR5P1C = vmlaq_laneq_f32(vecR5P1C, vecC5P1B, vecR5P1A, 1);
                    vecR5P1C = vmlaq_laneq_f32(vecR5P1C, vecC6P1B, vecR5P1A, 2);
                    vecR5P1C = vmlaq_laneq_f32(vecR5P1C, vecC7P1B, vecR5P1A, 3);

                    vecR6P0C = vmlaq_laneq_f32(vecR6P0C, vecC0P0B, vecR6P0A, 0);
                    vecR6P0C = vmlaq_laneq_f32(vecR6P0C, vecC1P0B, vecR6P0A, 1);
                    vecR6P0C = vmlaq_laneq_f32(vecR6P0C, vecC2P0B, vecR6P0A, 2);
                    vecR6P0C = vmlaq_laneq_f32(vecR6P0C, vecC3P0B, vecR6P0A, 3);
                    vecR6P0C = vmlaq_laneq_f32(vecR6P0C, vecC4P0B, vecR6P1A, 0);
                    vecR6P0C = vmlaq_laneq_f32(vecR6P0C, vecC5P0B, vecR6P1A, 1);
                    vecR6P0C = vmlaq_laneq_f32(vecR6P0C, vecC6P0B, vecR6P1A, 2);
                    vecR6P0C = vmlaq_laneq_f32(vecR6P0C, vecC7P0B, vecR6P1A, 3);
                    vecR6P1C = vmlaq_laneq_f32(vecR6P1C, vecC0P1B, vecR6P0A, 0);
                    vecR6P1C = vmlaq_laneq_f32(vecR6P1C, vecC1P1B, vecR6P0A, 1);
                    vecR6P1C = vmlaq_laneq_f32(vecR6P1C, vecC2P1B, vecR6P0A, 2);
                    vecR6P1C = vmlaq_laneq_f32(vecR6P1C, vecC3P1B, vecR6P0A, 3);
                    vecR6P1C = vmlaq_laneq_f32(vecR6P1C, vecC4P1B, vecR6P1A, 0);
                    vecR6P1C = vmlaq_laneq_f32(vecR6P1C, vecC5P1B, vecR6P1A, 1);
                    vecR6P1C = vmlaq_laneq_f32(vecR6P1C, vecC6P1B, vecR6P1A, 2);
                    vecR6P1C = vmlaq_laneq_f32(vecR6P1C, vecC7P1B, vecR6P1A, 3);

                    vecR7P0C = vmlaq_laneq_f32(vecR7P0C, vecC0P0B, vecR7P0A, 0);
                    vecR7P0C = vmlaq_laneq_f32(vecR7P0C, vecC1P0B, vecR7P0A, 1);
                    vecR7P0C = vmlaq_laneq_f32(vecR7P0C, vecC2P0B, vecR7P0A, 2);
                    vecR7P0C = vmlaq_laneq_f32(vecR7P0C, vecC3P0B, vecR7P0A, 3);
                    vecR7P0C = vmlaq_laneq_f32(vecR7P0C, vecC4P0B, vecR7P1A, 0);
                    vecR7P0C = vmlaq_laneq_f32(vecR7P0C, vecC5P0B, vecR7P1A, 1);
                    vecR7P0C = vmlaq_laneq_f32(vecR7P0C, vecC6P0B, vecR7P1A, 2);
                    vecR7P0C = vmlaq_laneq_f32(vecR7P0C, vecC7P0B, vecR7P1A, 3);
                    vecR7P1C = vmlaq_laneq_f32(vecR7P1C, vecC0P1B, vecR7P0A, 0);
                    vecR7P1C = vmlaq_laneq_f32(vecR7P1C, vecC1P1B, vecR7P0A, 1);
                    vecR7P1C = vmlaq_laneq_f32(vecR7P1C, vecC2P1B, vecR7P0A, 2);
                    vecR7P1C = vmlaq_laneq_f32(vecR7P1C, vecC3P1B, vecR7P0A, 3);
                    vecR7P1C = vmlaq_laneq_f32(vecR7P1C, vecC4P1B, vecR7P1A, 0);
                    vecR7P1C = vmlaq_laneq_f32(vecR7P1C, vecC5P1B, vecR7P1A, 1);
                    vecR7P1C = vmlaq_laneq_f32(vecR7P1C, vecC6P1B, vecR7P1A, 2);
                    vecR7P1C = vmlaq_laneq_f32(vecR7P1C, vecC7P1B, vecR7P1A, 3);
#else
                    float r0c0A = pR0A[ki + 0];
                    float r0c1A = pR0A[ki + 1];
                    float r0c2A = pR0A[ki + 2];
                    float r0c3A = pR0A[ki + 3];
                    float r0c4A = pR0A[ki + 4];
                    float r0c5A = pR0A[ki + 5];
                    float r0c6A = pR0A[ki + 6];
                    float r0c7A = pR0A[ki + 7];
                    float r1c0A = pR1A[ki + 0];
                    float r1c1A = pR1A[ki + 1];
                    float r1c2A = pR1A[ki + 2];
                    float r1c3A = pR1A[ki + 3];
                    float r1c4A = pR1A[ki + 4];
                    float r1c5A = pR1A[ki + 5];
                    float r1c6A = pR1A[ki + 6];
                    float r1c7A = pR1A[ki + 7];
                    float r2c0A = pR2A[ki + 0];
                    float r2c1A = pR2A[ki + 1];
                    float r2c2A = pR2A[ki + 2];
                    float r2c3A = pR2A[ki + 3];
                    float r2c4A = pR2A[ki + 4];
                    float r2c5A = pR2A[ki + 5];
                    float r2c6A = pR2A[ki + 6];
                    float r2c7A = pR2A[ki + 7];
                    float r3c0A = pR3A[ki + 0];
                    float r3c1A = pR3A[ki + 1];
                    float r3c2A = pR3A[ki + 2];
                    float r3c3A = pR3A[ki + 3];
                    float r3c4A = pR3A[ki + 4];
                    float r3c5A = pR3A[ki + 5];
                    float r3c6A = pR3A[ki + 6];
                    float r3c7A = pR3A[ki + 7];
                    float r4c0A = pR4A[ki + 0];
                    float r4c1A = pR4A[ki + 1];
                    float r4c2A = pR4A[ki + 2];
                    float r4c3A = pR4A[ki + 3];
                    float r4c4A = pR4A[ki + 4];
                    float r4c5A = pR4A[ki + 5];
                    float r4c6A = pR4A[ki + 6];
                    float r4c7A = pR4A[ki + 7];
                    float r5c0A = pR5A[ki + 0];
                    float r5c1A = pR5A[ki + 1];
                    float r5c2A = pR5A[ki + 2];
                    float r5c3A = pR5A[ki + 3];
                    float r5c4A = pR5A[ki + 4];
                    float r5c5A = pR5A[ki + 5];
                    float r5c6A = pR5A[ki + 6];
                    float r5c7A = pR5A[ki + 7];
                    float r6c0A = pR6A[ki + 0];
                    float r6c1A = pR6A[ki + 1];
                    float r6c2A = pR6A[ki + 2];
                    float r6c3A = pR6A[ki + 3];
                    float r6c4A = pR6A[ki + 4];
                    float r6c5A = pR6A[ki + 5];
                    float r6c6A = pR6A[ki + 6];
                    float r6c7A = pR6A[ki + 7];
                    float r7c0A = pR7A[ki + 0];
                    float r7c1A = pR7A[ki + 1];
                    float r7c2A = pR7A[ki + 2];
                    float r7c3A = pR7A[ki + 3];
                    float r7c4A = pR7A[ki + 4];
                    float r7c5A = pR7A[ki + 5];
                    float r7c6A = pR7A[ki + 6];
                    float r7c7A = pR7A[ki + 7];

                    float r0c0B = pR0B[ki + 0];
                    float r0c1B = pR0B[ki + 1];
                    float r0c2B = pR0B[ki + 2];
                    float r0c3B = pR0B[ki + 3];
                    float r0c4B = pR0B[ki + 4];
                    float r0c5B = pR0B[ki + 5];
                    float r0c6B = pR0B[ki + 6];
                    float r0c7B = pR0B[ki + 7];
                    float r1c0B = pR1B[ki + 0];
                    float r1c1B = pR1B[ki + 1];
                    float r1c2B = pR1B[ki + 2];
                    float r1c3B = pR1B[ki + 3];
                    float r1c4B = pR1B[ki + 4];
                    float r1c5B = pR1B[ki + 5];
                    float r1c6B = pR1B[ki + 6];
                    float r1c7B = pR1B[ki + 7];
                    float r2c0B = pR2B[ki + 0];
                    float r2c1B = pR2B[ki + 1];
                    float r2c2B = pR2B[ki + 2];
                    float r2c3B = pR2B[ki + 3];
                    float r2c4B = pR2B[ki + 4];
                    float r2c5B = pR2B[ki + 5];
                    float r2c6B = pR2B[ki + 6];
                    float r2c7B = pR2B[ki + 7];
                    float r3c0B = pR3B[ki + 0];
                    float r3c1B = pR3B[ki + 1];
                    float r3c2B = pR3B[ki + 2];
                    float r3c3B = pR3B[ki + 3];
                    float r3c4B = pR3B[ki + 4];
                    float r3c5B = pR3B[ki + 5];
                    float r3c6B = pR3B[ki + 6];
                    float r3c7B = pR3B[ki + 7];
                    float r4c0B = pR4B[ki + 0];
                    float r4c1B = pR4B[ki + 1];
                    float r4c2B = pR4B[ki + 2];
                    float r4c3B = pR4B[ki + 3];
                    float r4c4B = pR4B[ki + 4];
                    float r4c5B = pR4B[ki + 5];
                    float r4c6B = pR4B[ki + 6];
                    float r4c7B = pR4B[ki + 7];
                    float r5c0B = pR5B[ki + 0];
                    float r5c1B = pR5B[ki + 1];
                    float r5c2B = pR5B[ki + 2];
                    float r5c3B = pR5B[ki + 3];
                    float r5c4B = pR5B[ki + 4];
                    float r5c5B = pR5B[ki + 5];
                    float r5c6B = pR5B[ki + 6];
                    float r5c7B = pR5B[ki + 7];
                    float r6c0B = pR6B[ki + 0];
                    float r6c1B = pR6B[ki + 1];
                    float r6c2B = pR6B[ki + 2];
                    float r6c3B = pR6B[ki + 3];
                    float r6c4B = pR6B[ki + 4];
                    float r6c5B = pR6B[ki + 5];
                    float r6c6B = pR6B[ki + 6];
                    float r6c7B = pR6B[ki + 7];
                    float r7c0B = pR7B[ki + 0];
                    float r7c1B = pR7B[ki + 1];
                    float r7c2B = pR7B[ki + 2];
                    float r7c3B = pR7B[ki + 3];
                    float r7c4B = pR7B[ki + 4];
                    float r7c5B = pR7B[ki + 5];
                    float r7c6B = pR7B[ki + 6];
                    float r7c7B = pR7B[ki + 7];

                    r0c0C += r0c0A * r0c0B;
                    r0c0C += r0c1A * r0c1B;
                    r0c0C += r0c2A * r0c2B;
                    r0c0C += r0c3A * r0c3B;
                    r0c0C += r0c4A * r0c4B;
                    r0c0C += r0c5A * r0c5B;
                    r0c0C += r0c6A * r0c6B;
                    r0c0C += r0c7A * r0c7B;
                    r0c1C += r0c0A * r1c0B;
                    r0c1C += r0c1A * r1c1B;
                    r0c1C += r0c2A * r1c2B;
                    r0c1C += r0c3A * r1c3B;
                    r0c1C += r0c4A * r1c4B;
                    r0c1C += r0c5A * r1c5B;
                    r0c1C += r0c6A * r1c6B;
                    r0c1C += r0c7A * r1c7B;
                    r0c2C += r0c0A * r2c0B;
                    r0c2C += r0c1A * r2c1B;
                    r0c2C += r0c2A * r2c2B;
                    r0c2C += r0c3A * r2c3B;
                    r0c2C += r0c4A * r2c4B;
                    r0c2C += r0c5A * r2c5B;
                    r0c2C += r0c6A * r2c6B;
                    r0c2C += r0c7A * r2c7B;
                    r0c3C += r0c0A * r3c0B;
                    r0c3C += r0c1A * r3c1B;
                    r0c3C += r0c2A * r3c2B;
                    r0c3C += r0c3A * r3c3B;
                    r0c3C += r0c4A * r3c4B;
                    r0c3C += r0c5A * r3c5B;
                    r0c3C += r0c6A * r3c6B;
                    r0c3C += r0c7A * r3c7B;
                    r0c4C += r0c0A * r4c0B;
                    r0c4C += r0c1A * r4c1B;
                    r0c4C += r0c2A * r4c2B;
                    r0c4C += r0c3A * r4c3B;
                    r0c4C += r0c4A * r4c4B;
                    r0c4C += r0c5A * r4c5B;
                    r0c4C += r0c6A * r4c6B;
                    r0c4C += r0c7A * r4c7B;
                    r0c5C += r0c0A * r5c0B;
                    r0c5C += r0c1A * r5c1B;
                    r0c5C += r0c2A * r5c2B;
                    r0c5C += r0c3A * r5c3B;
                    r0c5C += r0c4A * r5c4B;
                    r0c5C += r0c5A * r5c5B;
                    r0c5C += r0c6A * r5c6B;
                    r0c5C += r0c7A * r5c7B;
                    r0c6C += r0c0A * r6c0B;
                    r0c6C += r0c1A * r6c1B;
                    r0c6C += r0c2A * r6c2B;
                    r0c6C += r0c3A * r6c3B;
                    r0c6C += r0c4A * r6c4B;
                    r0c6C += r0c5A * r6c5B;
                    r0c6C += r0c6A * r6c6B;
                    r0c6C += r0c7A * r6c7B;
                    r0c7C += r0c0A * r7c0B;
                    r0c7C += r0c1A * r7c1B;
                    r0c7C += r0c2A * r7c2B;
                    r0c7C += r0c3A * r7c3B;
                    r0c7C += r0c4A * r7c4B;
                    r0c7C += r0c5A * r7c5B;
                    r0c7C += r0c6A * r7c6B;
                    r0c7C += r0c7A * r7c7B;

                    r1c0C += r1c0A * r0c0B;
                    r1c0C += r1c1A * r0c1B;
                    r1c0C += r1c2A * r0c2B;
                    r1c0C += r1c3A * r0c3B;
                    r1c0C += r1c4A * r0c4B;
                    r1c0C += r1c5A * r0c5B;
                    r1c0C += r1c6A * r0c6B;
                    r1c0C += r1c7A * r0c7B;
                    r1c1C += r1c0A * r1c0B;
                    r1c1C += r1c1A * r1c1B;
                    r1c1C += r1c2A * r1c2B;
                    r1c1C += r1c3A * r1c3B;
                    r1c1C += r1c4A * r1c4B;
                    r1c1C += r1c5A * r1c5B;
                    r1c1C += r1c6A * r1c6B;
                    r1c1C += r1c7A * r1c7B;
                    r1c2C += r1c0A * r2c0B;
                    r1c2C += r1c1A * r2c1B;
                    r1c2C += r1c2A * r2c2B;
                    r1c2C += r1c3A * r2c3B;
                    r1c2C += r1c4A * r2c4B;
                    r1c2C += r1c5A * r2c5B;
                    r1c2C += r1c6A * r2c6B;
                    r1c2C += r1c7A * r2c7B;
                    r1c3C += r1c0A * r3c0B;
                    r1c3C += r1c1A * r3c1B;
                    r1c3C += r1c2A * r3c2B;
                    r1c3C += r1c3A * r3c3B;
                    r1c3C += r1c4A * r3c4B;
                    r1c3C += r1c5A * r3c5B;
                    r1c3C += r1c6A * r3c6B;
                    r1c3C += r1c7A * r3c7B;
                    r1c4C += r1c0A * r4c0B;
                    r1c4C += r1c1A * r4c1B;
                    r1c4C += r1c2A * r4c2B;
                    r1c4C += r1c3A * r4c3B;
                    r1c4C += r1c4A * r4c4B;
                    r1c4C += r1c5A * r4c5B;
                    r1c4C += r1c6A * r4c6B;
                    r1c4C += r1c7A * r4c7B;
                    r1c5C += r1c0A * r5c0B;
                    r1c5C += r1c1A * r5c1B;
                    r1c5C += r1c2A * r5c2B;
                    r1c5C += r1c3A * r5c3B;
                    r1c5C += r1c4A * r5c4B;
                    r1c5C += r1c5A * r5c5B;
                    r1c5C += r1c6A * r5c6B;
                    r1c5C += r1c7A * r5c7B;
                    r1c6C += r1c0A * r6c0B;
                    r1c6C += r1c1A * r6c1B;
                    r1c6C += r1c2A * r6c2B;
                    r1c6C += r1c3A * r6c3B;
                    r1c6C += r1c4A * r6c4B;
                    r1c6C += r1c5A * r6c5B;
                    r1c6C += r1c6A * r6c6B;
                    r1c6C += r1c7A * r6c7B;
                    r1c7C += r1c0A * r7c0B;
                    r1c7C += r1c1A * r7c1B;
                    r1c7C += r1c2A * r7c2B;
                    r1c7C += r1c3A * r7c3B;
                    r1c7C += r1c4A * r7c4B;
                    r1c7C += r1c5A * r7c5B;
                    r1c7C += r1c6A * r7c6B;
                    r1c7C += r1c7A * r7c7B;

                    r2c0C += r2c0A * r0c0B;
                    r2c0C += r2c1A * r0c1B;
                    r2c0C += r2c2A * r0c2B;
                    r2c0C += r2c3A * r0c3B;
                    r2c0C += r2c4A * r0c4B;
                    r2c0C += r2c5A * r0c5B;
                    r2c0C += r2c6A * r0c6B;
                    r2c0C += r2c7A * r0c7B;
                    r2c1C += r2c0A * r1c0B;
                    r2c1C += r2c1A * r1c1B;
                    r2c1C += r2c2A * r1c2B;
                    r2c1C += r2c3A * r1c3B;
                    r2c1C += r2c4A * r1c4B;
                    r2c1C += r2c5A * r1c5B;
                    r2c1C += r2c6A * r1c6B;
                    r2c1C += r2c7A * r1c7B;
                    r2c2C += r2c0A * r2c0B;
                    r2c2C += r2c1A * r2c1B;
                    r2c2C += r2c2A * r2c2B;
                    r2c2C += r2c3A * r2c3B;
                    r2c2C += r2c4A * r2c4B;
                    r2c2C += r2c5A * r2c5B;
                    r2c2C += r2c6A * r2c6B;
                    r2c2C += r2c7A * r2c7B;
                    r2c3C += r2c0A * r3c0B;
                    r2c3C += r2c1A * r3c1B;
                    r2c3C += r2c2A * r3c2B;
                    r2c3C += r2c3A * r3c3B;
                    r2c3C += r2c4A * r3c4B;
                    r2c3C += r2c5A * r3c5B;
                    r2c3C += r2c6A * r3c6B;
                    r2c3C += r2c7A * r3c7B;
                    r2c4C += r2c0A * r4c0B;
                    r2c4C += r2c1A * r4c1B;
                    r2c4C += r2c2A * r4c2B;
                    r2c4C += r2c3A * r4c3B;
                    r2c4C += r2c4A * r4c4B;
                    r2c4C += r2c5A * r4c5B;
                    r2c4C += r2c6A * r4c6B;
                    r2c4C += r2c7A * r4c7B;
                    r2c5C += r2c0A * r5c0B;
                    r2c5C += r2c1A * r5c1B;
                    r2c5C += r2c2A * r5c2B;
                    r2c5C += r2c3A * r5c3B;
                    r2c5C += r2c4A * r5c4B;
                    r2c5C += r2c5A * r5c5B;
                    r2c5C += r2c6A * r5c6B;
                    r2c5C += r2c7A * r5c7B;
                    r2c6C += r2c0A * r6c0B;
                    r2c6C += r2c1A * r6c1B;
                    r2c6C += r2c2A * r6c2B;
                    r2c6C += r2c3A * r6c3B;
                    r2c6C += r2c4A * r6c4B;
                    r2c6C += r2c5A * r6c5B;
                    r2c6C += r2c6A * r6c6B;
                    r2c6C += r2c7A * r6c7B;
                    r2c7C += r2c0A * r7c0B;
                    r2c7C += r2c1A * r7c1B;
                    r2c7C += r2c2A * r7c2B;
                    r2c7C += r2c3A * r7c3B;
                    r2c7C += r2c4A * r7c4B;
                    r2c7C += r2c5A * r7c5B;
                    r2c7C += r2c6A * r7c6B;
                    r2c7C += r2c7A * r7c7B;

                    r3c0C += r3c0A * r0c0B;
                    r3c0C += r3c1A * r0c1B;
                    r3c0C += r3c2A * r0c2B;
                    r3c0C += r3c3A * r0c3B;
                    r3c0C += r3c4A * r0c4B;
                    r3c0C += r3c5A * r0c5B;
                    r3c0C += r3c6A * r0c6B;
                    r3c0C += r3c7A * r0c7B;
                    r3c1C += r3c0A * r1c0B;
                    r3c1C += r3c1A * r1c1B;
                    r3c1C += r3c2A * r1c2B;
                    r3c1C += r3c3A * r1c3B;
                    r3c1C += r3c4A * r1c4B;
                    r3c1C += r3c5A * r1c5B;
                    r3c1C += r3c6A * r1c6B;
                    r3c1C += r3c7A * r1c7B;
                    r3c2C += r3c0A * r2c0B;
                    r3c2C += r3c1A * r2c1B;
                    r3c2C += r3c2A * r2c2B;
                    r3c2C += r3c3A * r2c3B;
                    r3c2C += r3c4A * r2c4B;
                    r3c2C += r3c5A * r2c5B;
                    r3c2C += r3c6A * r2c6B;
                    r3c2C += r3c7A * r2c7B;
                    r3c3C += r3c0A * r3c0B;
                    r3c3C += r3c1A * r3c1B;
                    r3c3C += r3c2A * r3c2B;
                    r3c3C += r3c3A * r3c3B;
                    r3c3C += r3c4A * r3c4B;
                    r3c3C += r3c5A * r3c5B;
                    r3c3C += r3c6A * r3c6B;
                    r3c3C += r3c7A * r3c7B;
                    r3c4C += r3c0A * r4c0B;
                    r3c4C += r3c1A * r4c1B;
                    r3c4C += r3c2A * r4c2B;
                    r3c4C += r3c3A * r4c3B;
                    r3c4C += r3c4A * r4c4B;
                    r3c4C += r3c5A * r4c5B;
                    r3c4C += r3c6A * r4c6B;
                    r3c4C += r3c7A * r4c7B;
                    r3c5C += r3c0A * r5c0B;
                    r3c5C += r3c1A * r5c1B;
                    r3c5C += r3c2A * r5c2B;
                    r3c5C += r3c3A * r5c3B;
                    r3c5C += r3c4A * r5c4B;
                    r3c5C += r3c5A * r5c5B;
                    r3c5C += r3c6A * r5c6B;
                    r3c5C += r3c7A * r5c7B;
                    r3c6C += r3c0A * r6c0B;
                    r3c6C += r3c1A * r6c1B;
                    r3c6C += r3c2A * r6c2B;
                    r3c6C += r3c3A * r6c3B;
                    r3c6C += r3c4A * r6c4B;
                    r3c6C += r3c5A * r6c5B;
                    r3c6C += r3c6A * r6c6B;
                    r3c6C += r3c7A * r6c7B;
                    r3c7C += r3c0A * r7c0B;
                    r3c7C += r3c1A * r7c1B;
                    r3c7C += r3c2A * r7c2B;
                    r3c7C += r3c3A * r7c3B;
                    r3c7C += r3c4A * r7c4B;
                    r3c7C += r3c5A * r7c5B;
                    r3c7C += r3c6A * r7c6B;
                    r3c7C += r3c7A * r7c7B;

                    r4c0C += r4c0A * r0c0B;
                    r4c0C += r4c1A * r0c1B;
                    r4c0C += r4c2A * r0c2B;
                    r4c0C += r4c3A * r0c3B;
                    r4c0C += r4c4A * r0c4B;
                    r4c0C += r4c5A * r0c5B;
                    r4c0C += r4c6A * r0c6B;
                    r4c0C += r4c7A * r0c7B;
                    r4c1C += r4c0A * r1c0B;
                    r4c1C += r4c1A * r1c1B;
                    r4c1C += r4c2A * r1c2B;
                    r4c1C += r4c3A * r1c3B;
                    r4c1C += r4c4A * r1c4B;
                    r4c1C += r4c5A * r1c5B;
                    r4c1C += r4c6A * r1c6B;
                    r4c1C += r4c7A * r1c7B;
                    r4c2C += r4c0A * r2c0B;
                    r4c2C += r4c1A * r2c1B;
                    r4c2C += r4c2A * r2c2B;
                    r4c2C += r4c3A * r2c3B;
                    r4c2C += r4c4A * r2c4B;
                    r4c2C += r4c5A * r2c5B;
                    r4c2C += r4c6A * r2c6B;
                    r4c2C += r4c7A * r2c7B;
                    r4c3C += r4c0A * r3c0B;
                    r4c3C += r4c1A * r3c1B;
                    r4c3C += r4c2A * r3c2B;
                    r4c3C += r4c3A * r3c3B;
                    r4c3C += r4c4A * r3c4B;
                    r4c3C += r4c5A * r3c5B;
                    r4c3C += r4c6A * r3c6B;
                    r4c3C += r4c7A * r3c7B;
                    r4c4C += r4c0A * r4c0B;
                    r4c4C += r4c1A * r4c1B;
                    r4c4C += r4c2A * r4c2B;
                    r4c4C += r4c3A * r4c3B;
                    r4c4C += r4c4A * r4c4B;
                    r4c4C += r4c5A * r4c5B;
                    r4c4C += r4c6A * r4c6B;
                    r4c4C += r4c7A * r4c7B;
                    r4c5C += r4c0A * r5c0B;
                    r4c5C += r4c1A * r5c1B;
                    r4c5C += r4c2A * r5c2B;
                    r4c5C += r4c3A * r5c3B;
                    r4c5C += r4c4A * r5c4B;
                    r4c5C += r4c5A * r5c5B;
                    r4c5C += r4c6A * r5c6B;
                    r4c5C += r4c7A * r5c7B;
                    r4c6C += r4c0A * r6c0B;
                    r4c6C += r4c1A * r6c1B;
                    r4c6C += r4c2A * r6c2B;
                    r4c6C += r4c3A * r6c3B;
                    r4c6C += r4c4A * r6c4B;
                    r4c6C += r4c5A * r6c5B;
                    r4c6C += r4c6A * r6c6B;
                    r4c6C += r4c7A * r6c7B;
                    r4c7C += r4c0A * r7c0B;
                    r4c7C += r4c1A * r7c1B;
                    r4c7C += r4c2A * r7c2B;
                    r4c7C += r4c3A * r7c3B;
                    r4c7C += r4c4A * r7c4B;
                    r4c7C += r4c5A * r7c5B;
                    r4c7C += r4c6A * r7c6B;
                    r4c7C += r4c7A * r7c7B;

                    r5c0C += r5c0A * r0c0B;
                    r5c0C += r5c1A * r0c1B;
                    r5c0C += r5c2A * r0c2B;
                    r5c0C += r5c3A * r0c3B;
                    r5c0C += r5c4A * r0c4B;
                    r5c0C += r5c5A * r0c5B;
                    r5c0C += r5c6A * r0c6B;
                    r5c0C += r5c7A * r0c7B;
                    r5c1C += r5c0A * r1c0B;
                    r5c1C += r5c1A * r1c1B;
                    r5c1C += r5c2A * r1c2B;
                    r5c1C += r5c3A * r1c3B;
                    r5c1C += r5c4A * r1c4B;
                    r5c1C += r5c5A * r1c5B;
                    r5c1C += r5c6A * r1c6B;
                    r5c1C += r5c7A * r1c7B;
                    r5c2C += r5c0A * r2c0B;
                    r5c2C += r5c1A * r2c1B;
                    r5c2C += r5c2A * r2c2B;
                    r5c2C += r5c3A * r2c3B;
                    r5c2C += r5c4A * r2c4B;
                    r5c2C += r5c5A * r2c5B;
                    r5c2C += r5c6A * r2c6B;
                    r5c2C += r5c7A * r2c7B;
                    r5c3C += r5c0A * r3c0B;
                    r5c3C += r5c1A * r3c1B;
                    r5c3C += r5c2A * r3c2B;
                    r5c3C += r5c3A * r3c3B;
                    r5c3C += r5c4A * r3c4B;
                    r5c3C += r5c5A * r3c5B;
                    r5c3C += r5c6A * r3c6B;
                    r5c3C += r5c7A * r3c7B;
                    r5c4C += r5c0A * r4c0B;
                    r5c4C += r5c1A * r4c1B;
                    r5c4C += r5c2A * r4c2B;
                    r5c4C += r5c3A * r4c3B;
                    r5c4C += r5c4A * r4c4B;
                    r5c4C += r5c5A * r4c5B;
                    r5c4C += r5c6A * r4c6B;
                    r5c4C += r5c7A * r4c7B;
                    r5c5C += r5c0A * r5c0B;
                    r5c5C += r5c1A * r5c1B;
                    r5c5C += r5c2A * r5c2B;
                    r5c5C += r5c3A * r5c3B;
                    r5c5C += r5c4A * r5c4B;
                    r5c5C += r5c5A * r5c5B;
                    r5c5C += r5c6A * r5c6B;
                    r5c5C += r5c7A * r5c7B;
                    r5c6C += r5c0A * r6c0B;
                    r5c6C += r5c1A * r6c1B;
                    r5c6C += r5c2A * r6c2B;
                    r5c6C += r5c3A * r6c3B;
                    r5c6C += r5c4A * r6c4B;
                    r5c6C += r5c5A * r6c5B;
                    r5c6C += r5c6A * r6c6B;
                    r5c6C += r5c7A * r6c7B;
                    r5c7C += r5c0A * r7c0B;
                    r5c7C += r5c1A * r7c1B;
                    r5c7C += r5c2A * r7c2B;
                    r5c7C += r5c3A * r7c3B;
                    r5c7C += r5c4A * r7c4B;
                    r5c7C += r5c5A * r7c5B;
                    r5c7C += r5c6A * r7c6B;
                    r5c7C += r5c7A * r7c7B;

                    r6c0C += r6c0A * r0c0B;
                    r6c0C += r6c1A * r0c1B;
                    r6c0C += r6c2A * r0c2B;
                    r6c0C += r6c3A * r0c3B;
                    r6c0C += r6c4A * r0c4B;
                    r6c0C += r6c5A * r0c5B;
                    r6c0C += r6c6A * r0c6B;
                    r6c0C += r6c7A * r0c7B;
                    r6c1C += r6c0A * r1c0B;
                    r6c1C += r6c1A * r1c1B;
                    r6c1C += r6c2A * r1c2B;
                    r6c1C += r6c3A * r1c3B;
                    r6c1C += r6c4A * r1c4B;
                    r6c1C += r6c5A * r1c5B;
                    r6c1C += r6c6A * r1c6B;
                    r6c1C += r6c7A * r1c7B;
                    r6c2C += r6c0A * r2c0B;
                    r6c2C += r6c1A * r2c1B;
                    r6c2C += r6c2A * r2c2B;
                    r6c2C += r6c3A * r2c3B;
                    r6c2C += r6c4A * r2c4B;
                    r6c2C += r6c5A * r2c5B;
                    r6c2C += r6c6A * r2c6B;
                    r6c2C += r6c7A * r2c7B;
                    r6c3C += r6c0A * r3c0B;
                    r6c3C += r6c1A * r3c1B;
                    r6c3C += r6c2A * r3c2B;
                    r6c3C += r6c3A * r3c3B;
                    r6c3C += r6c4A * r3c4B;
                    r6c3C += r6c5A * r3c5B;
                    r6c3C += r6c6A * r3c6B;
                    r6c3C += r6c7A * r3c7B;
                    r6c4C += r6c0A * r4c0B;
                    r6c4C += r6c1A * r4c1B;
                    r6c4C += r6c2A * r4c2B;
                    r6c4C += r6c3A * r4c3B;
                    r6c4C += r6c4A * r4c4B;
                    r6c4C += r6c5A * r4c5B;
                    r6c4C += r6c6A * r4c6B;
                    r6c4C += r6c7A * r4c7B;
                    r6c5C += r6c0A * r5c0B;
                    r6c5C += r6c1A * r5c1B;
                    r6c5C += r6c2A * r5c2B;
                    r6c5C += r6c3A * r5c3B;
                    r6c5C += r6c4A * r5c4B;
                    r6c5C += r6c5A * r5c5B;
                    r6c5C += r6c6A * r5c6B;
                    r6c5C += r6c7A * r5c7B;
                    r6c6C += r6c0A * r6c0B;
                    r6c6C += r6c1A * r6c1B;
                    r6c6C += r6c2A * r6c2B;
                    r6c6C += r6c3A * r6c3B;
                    r6c6C += r6c4A * r6c4B;
                    r6c6C += r6c5A * r6c5B;
                    r6c6C += r6c6A * r6c6B;
                    r6c6C += r6c7A * r6c7B;
                    r6c7C += r6c0A * r7c0B;
                    r6c7C += r6c1A * r7c1B;
                    r6c7C += r6c2A * r7c2B;
                    r6c7C += r6c3A * r7c3B;
                    r6c7C += r6c4A * r7c4B;
                    r6c7C += r6c5A * r7c5B;
                    r6c7C += r6c6A * r7c6B;
                    r6c7C += r6c7A * r7c7B;

                    r7c0C += r7c0A * r0c0B;
                    r7c0C += r7c1A * r0c1B;
                    r7c0C += r7c2A * r0c2B;
                    r7c0C += r7c3A * r0c3B;
                    r7c0C += r7c4A * r0c4B;
                    r7c0C += r7c5A * r0c5B;
                    r7c0C += r7c6A * r0c6B;
                    r7c0C += r7c7A * r0c7B;
                    r7c1C += r7c0A * r1c0B;
                    r7c1C += r7c1A * r1c1B;
                    r7c1C += r7c2A * r1c2B;
                    r7c1C += r7c3A * r1c3B;
                    r7c1C += r7c4A * r1c4B;
                    r7c1C += r7c5A * r1c5B;
                    r7c1C += r7c6A * r1c6B;
                    r7c1C += r7c7A * r1c7B;
                    r7c2C += r7c0A * r2c0B;
                    r7c2C += r7c1A * r2c1B;
                    r7c2C += r7c2A * r2c2B;
                    r7c2C += r7c3A * r2c3B;
                    r7c2C += r7c4A * r2c4B;
                    r7c2C += r7c5A * r2c5B;
                    r7c2C += r7c6A * r2c6B;
                    r7c2C += r7c7A * r2c7B;
                    r7c3C += r7c0A * r3c0B;
                    r7c3C += r7c1A * r3c1B;
                    r7c3C += r7c2A * r3c2B;
                    r7c3C += r7c3A * r3c3B;
                    r7c3C += r7c4A * r3c4B;
                    r7c3C += r7c5A * r3c5B;
                    r7c3C += r7c6A * r3c6B;
                    r7c3C += r7c7A * r3c7B;
                    r7c4C += r7c0A * r4c0B;
                    r7c4C += r7c1A * r4c1B;
                    r7c4C += r7c2A * r4c2B;
                    r7c4C += r7c3A * r4c3B;
                    r7c4C += r7c4A * r4c4B;
                    r7c4C += r7c5A * r4c5B;
                    r7c4C += r7c6A * r4c6B;
                    r7c4C += r7c7A * r4c7B;
                    r7c5C += r7c0A * r5c0B;
                    r7c5C += r7c1A * r5c1B;
                    r7c5C += r7c2A * r5c2B;
                    r7c5C += r7c3A * r5c3B;
                    r7c5C += r7c4A * r5c4B;
                    r7c5C += r7c5A * r5c5B;
                    r7c5C += r7c6A * r5c6B;
                    r7c5C += r7c7A * r5c7B;
                    r7c6C += r7c0A * r6c0B;
                    r7c6C += r7c1A * r6c1B;
                    r7c6C += r7c2A * r6c2B;
                    r7c6C += r7c3A * r6c3B;
                    r7c6C += r7c4A * r6c4B;
                    r7c6C += r7c5A * r6c5B;
                    r7c6C += r7c6A * r6c6B;
                    r7c6C += r7c7A * r6c7B;
                    r7c7C += r7c0A * r7c0B;
                    r7c7C += r7c1A * r7c1B;
                    r7c7C += r7c2A * r7c2B;
                    r7c7C += r7c3A * r7c3B;
                    r7c7C += r7c4A * r7c4B;
                    r7c7C += r7c5A * r7c5B;
                    r7c7C += r7c6A * r7c6B;
                    r7c7C += r7c7A * r7c7B;
#endif
                }

                /* 处理K维度上的尾循环 */
                for (; ki < K; ki++) {
                    float r0ckA = pR0A[ki];
                    float r1ckA = pR1A[ki];
                    float r2ckA = pR2A[ki];
                    float r3ckA = pR3A[ki];
                    float r4ckA = pR4A[ki];
                    float r5ckA = pR5A[ki];
                    float r6ckA = pR6A[ki];
                    float r7ckA = pR7A[ki];
                    float r0ckB = pR0B[ki];
                    float r1ckB = pR1B[ki];
                    float r2ckB = pR2B[ki];
                    float r3ckB = pR3B[ki];
                    float r4ckB = pR4B[ki];
                    float r5ckB = pR5B[ki];
                    float r6ckB = pR6B[ki];
                    float r7ckB = pR7B[ki];
#if __ARM_NEON
                    float32x4_t vecCkP0B = {r0ckB, r1ckB, r2ckB, r3ckB};
                    float32x4_t vecCkP1B = {r4ckB, r5ckB, r6ckB, r7ckB};
                    vecR0P0C = vmlaq_n_f32(vecR0P0C, vecCkP0B, r0ckA);
                    vecR0P1C = vmlaq_n_f32(vecR0P1C, vecCkP1B, r0ckA);
                    vecR1P0C = vmlaq_n_f32(vecR1P0C, vecCkP0B, r1ckA);
                    vecR1P1C = vmlaq_n_f32(vecR1P1C, vecCkP1B, r1ckA);
                    vecR2P0C = vmlaq_n_f32(vecR2P0C, vecCkP0B, r2ckA);
                    vecR2P1C = vmlaq_n_f32(vecR2P1C, vecCkP1B, r2ckA);
                    vecR3P0C = vmlaq_n_f32(vecR3P0C, vecCkP0B, r3ckA);
                    vecR3P1C = vmlaq_n_f32(vecR3P1C, vecCkP1B, r3ckA);
#else
                    r0c0C += r0ckA * r0ckB;
                    r0c1C += r0ckA * r1ckB;
                    r0c2C += r0ckA * r2ckB;
                    r0c3C += r0ckA * r3ckB;
                    r1c0C += r1ckA * r0ckB;
                    r1c1C += r1ckA * r1ckB;
                    r1c2C += r1ckA * r2ckB;
                    r1c3C += r1ckA * r3ckB;
                    r2c0C += r2ckA * r0ckB;
                    r2c1C += r2ckA * r1ckB;
                    r2c2C += r2ckA * r2ckB;
                    r2c3C += r2ckA * r3ckB;
                    r3c0C += r3ckA * r0ckB;
                    r3c1C += r3ckA * r1ckB;
                    r3c2C += r3ckA * r2ckB;
                    r3c3C += r3ckA * r3ckB;
#endif
                }

#if __ARM_NEON
                vst1q_f32(pR0C + ni,     vecR0P0C);
                vst1q_f32(pR0C + ni + 4, vecR0P1C);
                vst1q_f32(pR1C + ni,     vecR1P0C);
                vst1q_f32(pR1C + ni + 4, vecR1P1C);
                vst1q_f32(pR2C + ni,     vecR2P0C);
                vst1q_f32(pR2C + ni + 4, vecR2P1C);
                vst1q_f32(pR3C + ni,     vecR3P0C);
                vst1q_f32(pR3C + ni + 4, vecR3P1C);
                vst1q_f32(pR4C + ni,     vecR4P0C);
                vst1q_f32(pR4C + ni + 4, vecR4P1C);
                vst1q_f32(pR5C + ni,     vecR5P0C);
                vst1q_f32(pR5C + ni + 4, vecR5P1C);
                vst1q_f32(pR6C + ni,     vecR6P0C);
                vst1q_f32(pR6C + ni + 4, vecR6P1C);
                vst1q_f32(pR7C + ni,     vecR7P0C);
                vst1q_f32(pR7C + ni + 4, vecR7P1C);
#else
                pR0C[ni + 0] = r0c0C;
                pR0C[ni + 1] = r0c1C;
                pR0C[ni + 2] = r0c2C;
                pR0C[ni + 3] = r0c3C;
                pR0C[ni + 4] = r0c4C;
                pR0C[ni + 5] = r0c5C;
                pR0C[ni + 6] = r0c6C;
                pR0C[ni + 7] = r0c7C;
                pR1C[ni + 0] = r1c0C;
                pR1C[ni + 1] = r1c1C;
                pR1C[ni + 2] = r1c2C;
                pR1C[ni + 3] = r1c3C;
                pR1C[ni + 4] = r1c4C;
                pR1C[ni + 5] = r1c5C;
                pR1C[ni + 6] = r1c6C;
                pR1C[ni + 7] = r1c7C;
                pR2C[ni + 0] = r2c0C;
                pR2C[ni + 1] = r2c1C;
                pR2C[ni + 2] = r2c2C;
                pR2C[ni + 3] = r2c3C;
                pR2C[ni + 4] = r2c4C;
                pR2C[ni + 5] = r2c5C;
                pR2C[ni + 6] = r2c6C;
                pR2C[ni + 7] = r2c7C;
                pR3C[ni + 0] = r3c0C;
                pR3C[ni + 1] = r3c1C;
                pR3C[ni + 2] = r3c2C;
                pR3C[ni + 3] = r3c3C;
                pR3C[ni + 4] = r3c4C;
                pR3C[ni + 5] = r3c5C;
                pR3C[ni + 6] = r3c6C;
                pR3C[ni + 7] = r3c7C;
                pR4C[ni + 0] = r4c0C;
                pR4C[ni + 1] = r4c1C;
                pR4C[ni + 2] = r4c2C;
                pR4C[ni + 3] = r4c3C;
                pR4C[ni + 4] = r4c4C;
                pR4C[ni + 5] = r4c5C;
                pR4C[ni + 6] = r4c6C;
                pR4C[ni + 7] = r4c7C;
                pR5C[ni + 0] = r5c0C;
                pR5C[ni + 1] = r5c1C;
                pR5C[ni + 2] = r5c2C;
                pR5C[ni + 3] = r5c3C;
                pR5C[ni + 4] = r5c4C;
                pR5C[ni + 5] = r5c5C;
                pR5C[ni + 6] = r5c6C;
                pR5C[ni + 7] = r5c7C;
                pR6C[ni + 0] = r6c0C;
                pR6C[ni + 1] = r6c1C;
                pR6C[ni + 2] = r6c2C;
                pR6C[ni + 3] = r6c3C;
                pR6C[ni + 4] = r6c4C;
                pR6C[ni + 5] = r6c5C;
                pR6C[ni + 6] = r6c6C;
                pR6C[ni + 7] = r6c7C;
                pR7C[ni + 0] = r7c0C;
                pR7C[ni + 1] = r7c1C;
                pR7C[ni + 2] = r7c2C;
                pR7C[ni + 3] = r7c3C;
                pR7C[ni + 4] = r7c4C;
                pR7C[ni + 5] = r7c5C;
                pR7C[ni + 6] = r7c6C;
                pR7C[ni + 7] = r7c7C;
#endif
            }

            /* 处理N维度上的尾循环 */
            for (; ni < N; ni++) {
                float r0C = 0.0f;
                float r1C = 0.0f;
                float r2C = 0.0f;
                float r3C = 0.0f;
                auto pRowB = pBatchB + ni * K;

                /* 在K维度上进行loop unrolling */
                size_t ki = 0;
                for (; ki < K - 3; ki += 4) {
                    float r0c0A = pR0A[ki + 0];
                    float r0c1A = pR0A[ki + 1];
                    float r0c2A = pR0A[ki + 2];
                    float r0c3A = pR0A[ki + 3];
                    float r1c0A = pR1A[ki + 0];
                    float r1c1A = pR1A[ki + 1];
                    float r1c2A = pR1A[ki + 2];
                    float r1c3A = pR1A[ki + 3];
                    float r2c0A = pR2A[ki + 0];
                    float r2c1A = pR2A[ki + 1];
                    float r2c2A = pR2A[ki + 2];
                    float r2c3A = pR2A[ki + 3];
                    float r3c0A = pR3A[ki + 0];
                    float r3c1A = pR3A[ki + 1];
                    float r3c2A = pR3A[ki + 2];
                    float r3c3A = pR3A[ki + 3];
                    float rnc0B = pRowB[ki + 0];
                    float rnc1B = pRowB[ki + 1];
                    float rnc2B = pRowB[ki + 2];
                    float rnc3B = pRowB[ki + 3];
                    r0C += r0c0A * rnc0B;
                    r0C += r0c1A * rnc1B;
                    r0C += r0c2A * rnc2B;
                    r0C += r0c3A * rnc3B;
                    r1C += r1c0A * rnc0B;
                    r1C += r1c1A * rnc1B;
                    r1C += r1c2A * rnc2B;
                    r1C += r1c3A * rnc3B;
                    r2C += r2c0A * rnc0B;
                    r2C += r2c1A * rnc1B;
                    r2C += r2c2A * rnc2B;
                    r2C += r2c3A * rnc3B;
                    r3C += r3c0A * rnc0B;
                    r3C += r3c1A * rnc1B;
                    r3C += r3c2A * rnc2B;
                    r3C += r3c3A * rnc3B;
                }

                /* 处理K维度上的尾循环 */
                for (; ki < K; ki++) {
                    float dataB = pRowB[ki];
                    r0C += pR0A[ki] * dataB;
                    r1C += pR1A[ki] * dataB;
                    r2C += pR2A[ki] * dataB;
                    r3C += pR3A[ki] * dataB;
                }

                pR0C[ni] = r0C;
                pR1C[ni] = r1C;
                pR2C[ni] = r2C;
                pR3C[ni] = r3C;
            }
        }

        /* 处理M维度上的尾循环 */
        for (; mi < M; mi++) {
            auto pRowC = pBatchC + mi * N;
            auto pRowA = pBatchA + mi * K;

            /* 在N维度上进行loop unrolling */
            size_t ni = 0;
            for (; ni < N - 3; ni += 4) {
                auto pR0B = pBatchB + (ni + 0) * K;
                auto pR1B = pBatchB + (ni + 1) * K;
                auto pR2B = pBatchB + (ni + 2) * K;
                auto pR3B = pBatchB + (ni + 3) * K;
#if __ARM_NEON
                float32x4_t vecRowC = vdupq_n_f32(0.0f);
#else
                float c0C = 0.0f;
                float c1C = 0.0f;
                float c2C = 0.0f;
                float c3C = 0.0f;
#endif

                /* 在K维度上进行loop unrolling */
                size_t ki = 0;
                for (; ki < K - 3; ki += 4) {
                    float rmc0A = pRowA[ki + 0];
                    float rmc1A = pRowA[ki + 1];
                    float rmc2A = pRowA[ki + 2];
                    float rmc3A = pRowA[ki + 3];
                    float r0c0B = pR0B[ki + 0];
                    float r0c1B = pR0B[ki + 1];
                    float r0c2B = pR0B[ki + 2];
                    float r0c3B = pR0B[ki + 3];
                    float r1c0B = pR1B[ki + 0];
                    float r1c1B = pR1B[ki + 1];
                    float r1c2B = pR1B[ki + 2];
                    float r1c3B = pR1B[ki + 3];
                    float r2c0B = pR2B[ki + 0];
                    float r2c1B = pR2B[ki + 1];
                    float r2c2B = pR2B[ki + 2];
                    float r2c3B = pR2B[ki + 3];
                    float r3c0B = pR3B[ki + 0];
                    float r3c1B = pR3B[ki + 1];
                    float r3c2B = pR3B[ki + 2];
                    float r3c3B = pR3B[ki + 3];
#if __ARM_NEON
                    float32x4_t vecC0B = {r0c0B, r1c0B, r2c0B, r3c0B};
                    float32x4_t vecC1B = {r0c1B, r1c1B, r2c1B, r3c1B};
                    float32x4_t vecC2B = {r0c2B, r1c2B, r2c2B, r3c2B};
                    float32x4_t vecC3B = {r0c3B, r1c3B, r2c3B, r3c3B};
                    vecRowC = vmlaq_n_f32(vecRowC, vecC0B, rmc0A);
                    vecRowC = vmlaq_n_f32(vecRowC, vecC1B, rmc1A);
                    vecRowC = vmlaq_n_f32(vecRowC, vecC2B, rmc2A);
                    vecRowC = vmlaq_n_f32(vecRowC, vecC3B, rmc3A);
#else
                    c0C += rmc0A * r0c0B;
                    c0C += rmc1A * r0c1B;
                    c0C += rmc2A * r0c2B;
                    c0C += rmc3A * r0c3B;
                    c1C += rmc0A * r1c0B;
                    c1C += rmc1A * r1c1B;
                    c1C += rmc2A * r1c2B;
                    c1C += rmc3A * r1c3B;
                    c2C += rmc0A * r2c0B;
                    c2C += rmc1A * r2c1B;
                    c2C += rmc2A * r2c2B;
                    c2C += rmc3A * r2c3B;
                    c3C += rmc0A * r3c0B;
                    c3C += rmc1A * r3c1B;
                    c3C += rmc2A * r3c2B;
                    c3C += rmc3A * r3c3B;
#endif
                }

                /* 处理K维度上的尾循环 */
                for (; ki < K; ki++) {
                    float dataA = pRowA[ki];
                    float r0ckB = pR0B[ki];
                    float r1ckB = pR1B[ki];
                    float r2ckB = pR2B[ki];
                    float r3ckB = pR3B[ki];
#if __ARM_NEON
                    float32x4_t vecColB = {r0ckB, r1ckB, r2ckB, r3ckB};
                    vecRowC = vmlaq_n_f32(vecRowC, vecColB, dataA);
#else
                    c0C += dataA * r0ckB;
                    c1C += dataA * r1ckB;
                    c2C += dataA * r2ckB;
                    c3C += dataA * r3ckB;
#endif
                }

#if __ARM_NEON
                vst1q_f32(pRowC + ni, vecRowC);
#else
                pRowC[ni + 0] = c0C;
                pRowC[ni + 1] = c1C;
                pRowC[ni + 2] = c2C;
                pRowC[ni + 3] = c3C;
#endif
            }

            /* 处理N维度上的尾循环 */
            for (; ni < N; ni++) {
                float sum = 0.0f;
                auto pRowB = pBatchB + ni * K;
                for (int ki = 0; ki < K; ki++) {
                    sum += pRowA[ki] * pRowB[ki];
                }
                pRowC[ni] = sum;
            }
        }
    }

    return 0;
}