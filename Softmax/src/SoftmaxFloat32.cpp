#include <math/MathFloat32.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cstring>
#include <cassert>

#if __ARM_NEON
#include <arm_neon.h>
#endif


#define RowMajorOrder
// #define ColumnMajorOrder


void SoftmaxFloat32(float *dst, const float *src, 
                    float *exp, float *max, float *sum, 
                    size_t height, size_t width, size_t channel)
{
    /**
     * Row-major order.
     * Only support to compute softmax along the channel dimension.
    */

    size_t ci = 0, cr = 0;
    /* Compute the maximum value */
    {
        for (size_t wi = 0; wi < width; ++wi) {
            size_t sw = wi * 4;
            float *maxW = max + sw;
            float *sumW = sum + sw;
#if __ARM_NEON
            vst1q_f32(sumW, vdupq_n_f32(0.0f));
            vst1q_f32(maxW, vdupq_n_f32(-FLT_MAX));
#else
            sumW[0] = 0.0f;
            sumW[1] = 0.0f;
            sumW[2] = 0.0f;
            sumW[3] = 0.0f;
            maxW[0] = -FLT_MAX;
            maxW[1] = -FLT_MAX;
            maxW[2] = -FLT_MAX;
            maxW[3] = -FLT_MAX;
#endif // __ARM_NEON
        }

        for (; ci < channel - 3; ci += 4) {
            float *expC = exp + ci * width;
            const float *inC = src + ci * width * height;
            for (size_t wi = 0; wi < width; ++wi) {
                size_t sw = wi * 4;
                float *maxW = max + sw;
                const float *inW = inC + sw;
#if __ARM_NEON
                float32x4_t vecInW = vld1q_f32(inW);
                float32x4_t vecMaxW = vld1q_f32(maxW);
                vecMaxW = vmaxnmq_f32(vecMaxW, vecInW);
                vecMaxW[0] = vmaxvq_f32(vecMaxW);
                vst1q_f32(maxW, vecMaxW);
#else
                float maxVal = maxW[0];
                for (size_t i = 0; i < 4; ++i) {
                    float maxTmp = std::fmax(inW[i], maxW[i]);
                    maxVal = std::fmax(maxVal, maxTmp);
                    maxW[i] = maxTmp;
                }
                maxW[0] = maxVal;
#endif // __ARM_NEON
            }
        }

        // Run remaining elements
        cr = channel - ci;
        if (cr != 0) {
            float *expC = exp + ci * width;
            const float *inC = src + ci * width * height;
            for (size_t wi = 0; wi < width; ++wi) {
                float *maxW = max + wi * 4;
                const float *inW = inC + wi * 4;
                float maxVal = maxW[0];
                for (size_t i = 0; i < 4; ++i) {
                    if (i < cr) {
                        maxVal = std::fmax(maxVal, inW[i]);
                    }
                }
                maxW[0] = maxVal;
            }
        }
    } // Compute the maximum value

    /* Compute the exponentials and sum value */
    {
        ci = 0;
        for (; ci < channel - 3; ci += 4) {
            float *expC = exp + ci * width;
            const float *inC = src + ci * width * height;
            for (size_t wi = 0; wi < width; ++wi) {
                size_t sw = wi * 4;
                float *maxW = max + sw;
                float *sumW = sum + sw;
                float *expW = expC + sw;
                const float *inW = inC + sw;
#if __ARM_NEON
                float32x4_t vecInW = vld1q_f32(inW);
                float32x4_t vecMaxW = vld1q_f32(maxW);
                float32x4_t vecMaxVal = vdupq_n_f32(vecMaxW[0]);
                Exp(expW, vsubq_f32(vecInW, vecMaxVal));
                float32x4_t vecExp = vld1q_f32(expW);
                float32x4_t vecSum = vld1q_f32(sumW);
                vst1q_f32(sumW, vaddq_f32(vecSum, vecExp));
#else
                float maxVal = maxW[0];
                for (size_t i = 0; i < 4; ++i) {
                    float tmp = std::exp(inW[i] - maxVal);
                    expW[i] = tmp;
                    sumW[i] += tmp;
                }
#endif // __ARM_NEON
            }
        }

        // Run remaining elements
        cr = channel - ci;
        if (cr != 0) {
            float *expC = exp + ci * width;
            const float *inC = src + ci * width * height;
            for (size_t wi = 0; wi < width; ++wi) {
                size_t sw = wi * 4;
                float *maxW = max + sw;
                float *sumW = sum + sw;
                float *expW = expC + sw;
                const float *inW = inC + sw;
                float maxVal = maxW[0];
                for (size_t i = 0; i < 4; ++i) {
                    if (i < cr) {
                        float tmp = std::exp(inW[i] - maxVal);
                        expW[i] = tmp;
                        sumW[i] += tmp;
                    }
                }
            }
        }
    } // Compute the exponentials and sum value

    /* Normalize exponentials */
    {
        ci = 0;
        for (; ci < channel - 3; ci += 4) {
            float *expC = exp + ci * width;
            float *outC = dst + ci * width * height;
            for (size_t wi = 0; wi < width; ++wi) {
                size_t sw = wi * 4;
                float *sumW = sum + sw;
                float *expW = expC + sw;
                float *outW = outC + sw;
#if __ARM_NEON
                float32x4_t vecExp = vld1q_f32(expW);
                float32x4_t vecSum = vld1q_f32(sumW);
                float32_t sumVal = vaddvq_f32(vecSum);
                float32_t recipSum = 1 / sumVal;
                vst1q_f32(outW, vmulq_n_f32(vecExp, recipSum));
#else
                float sumVal = sumW[0] + sumW[1] + sumW[2] + sumW[3];
                for (size_t i = 0; i < 4; ++i) {
                    outW[i] = expW[i] / sumVal;
                }
#endif // __ARM_NEON
            }
        }

        // Run remaining elements
        cr = channel - ci;
        if (cr != 0) {
            float *expC = exp + ci * width;
            float *outC = dst + ci * width * height;
            for (size_t wi = 0; wi < width; ++wi) {
                size_t sw = wi * 4;
                float *sumW = sum + sw;
                float *expW = expC + sw;
                float *outW = outC + sw;
                float sumVal = sumW[0] + sumW[1] + sumW[2] + sumW[3];
                for (size_t i = 0; i < 4; ++i) {
                    if (i < cr) {
                        outW[i] = expW[i] / sumVal;
                    }
                }
            }
        }
    } // Normalize exponentials

    return;
}


#ifdef ColumnMajorOrder
void SoftmaxFloat32(float *dst, const float *src, 
                    float *exp, float *max, float *sum, 
                    size_t height, size_t width, size_t channel)
{
    /**
     * Column-major order.
     * 
     * Only support to compute softmax along the last dimension
    */

    size_t ci = 0;
    for (; ci < channel - 7; ci += 8) {
        float *expC0 = exp + ci * width;
        float *expC1 = expC0 + width * 4;
        const float *inC0 = src + ci * height * width;
        const float *inC1 = inC0 + height * width * 4;

        size_t wi = 0;
        for (; wi < width - 7; wi += 8) {
#if __ARM_NEON
            float *sumW00 = sum + wi * 4;
            float *sumW01 = sum + (wi + 1) * 4;
            float *sumW02 = sum + (wi + 2) * 4;
            float *sumW03 = sum + (wi + 3) * 4;
            float *sumW04 = sum + (wi + 4) * 4;
            float *sumW05 = sum + (wi + 5) * 4;
            float *sumW06 = sum + (wi + 6) * 4;
            float *sumW07 = sum + (wi + 7) * 4;
            float32x4_t sumData00 = vld1q_f32(sumW00);
            float32x4_t sumData01 = vld1q_f32(sumW01);
            float32x4_t sumData02 = vld1q_f32(sumW02);
            float32x4_t sumData03 = vld1q_f32(sumW03);
            float32x4_t sumData04 = vld1q_f32(sumW04);
            float32x4_t sumData05 = vld1q_f32(sumW05);
            float32x4_t sumData06 = vld1q_f32(sumW06);
            float32x4_t sumData07 = vld1q_f32(sumW07);

            // row00
            float *expC0W00 = expC0 + wi * 4;
            float *expC0W01 = expC0 + (wi + 1) * 4;
            float *expC0W02 = expC0 + (wi + 2) * 4;
            float *expC0W03 = expC0 + (wi + 3) * 4;
            float *expC0W04 = expC0 + (wi + 4) * 4;
            float *expC0W05 = expC0 + (wi + 5) * 4;
            float *expC0W06 = expC0 + (wi + 6) * 4;
            float *expC0W07 = expC0 + (wi + 7) * 4;
            const float *inC0W00 = inC0 + wi * 4;
            const float *inC0W01 = inC0 + (wi + 1) * 4;
            const float *inC0W02 = inC0 + (wi + 2) * 4;
            const float *inC0W03 = inC0 + (wi + 3) * 4;
            const float *inC0W04 = inC0 + (wi + 4) * 4;
            const float *inC0W05 = inC0 + (wi + 5) * 4;
            const float *inC0W06 = inC0 + (wi + 6) * 4;
            const float *inC0W07 = inC0 + (wi + 7) * 4;
            Exp(expC0W00, inC0W00);
            Exp(expC0W01, inC0W01);
            Exp(expC0W02, inC0W02);
            Exp(expC0W03, inC0W03);
            Exp(expC0W04, inC0W04);
            Exp(expC0W05, inC0W05);
            Exp(expC0W06, inC0W06);
            Exp(expC0W07, inC0W07);
            float32x4_t expC0Data00 = vld1q_f32(expC0W00);
            float32x4_t expC0Data01 = vld1q_f32(expC0W01);
            float32x4_t expC0Data02 = vld1q_f32(expC0W02);
            float32x4_t expC0Data03 = vld1q_f32(expC0W03);
            float32x4_t expC0Data04 = vld1q_f32(expC0W04);
            float32x4_t expC0Data05 = vld1q_f32(expC0W05);
            float32x4_t expC0Data06 = vld1q_f32(expC0W06);
            float32x4_t expC0Data07 = vld1q_f32(expC0W07);
            sumData00 = vaddq_f32(sumData00, expC0Data00);
            sumData01 = vaddq_f32(sumData01, expC0Data01);
            sumData02 = vaddq_f32(sumData02, expC0Data02);
            sumData03 = vaddq_f32(sumData03, expC0Data03);
            sumData04 = vaddq_f32(sumData04, expC0Data04);
            sumData05 = vaddq_f32(sumData05, expC0Data05);
            sumData06 = vaddq_f32(sumData06, expC0Data06);
            sumData07 = vaddq_f32(sumData07, expC0Data07);
            // row01
            float *expC1W00 = expC1 + wi * 4;
            float *expC1W01 = expC1 + (wi + 1) * 4;
            float *expC1W02 = expC1 + (wi + 2) * 4;
            float *expC1W03 = expC1 + (wi + 3) * 4;
            float *expC1W04 = expC1 + (wi + 4) * 4;
            float *expC1W05 = expC1 + (wi + 5) * 4;
            float *expC1W06 = expC1 + (wi + 6) * 4;
            float *expC1W07 = expC1 + (wi + 7) * 4;
            const float *inC1W00 = inC1 + wi * 4;
            const float *inC1W01 = inC1 + (wi + 1) * 4;
            const float *inC1W02 = inC1 + (wi + 2) * 4;
            const float *inC1W03 = inC1 + (wi + 3) * 4;
            const float *inC1W04 = inC1 + (wi + 4) * 4;
            const float *inC1W05 = inC1 + (wi + 5) * 4;
            const float *inC1W06 = inC1 + (wi + 6) * 4;
            const float *inC1W07 = inC1 + (wi + 7) * 4;
            Exp(expC1W00, inC1W00);
            Exp(expC1W01, inC1W01);
            Exp(expC1W02, inC1W02);
            Exp(expC1W03, inC1W03);
            Exp(expC1W04, inC1W04);
            Exp(expC1W05, inC1W05);
            Exp(expC1W06, inC1W06);
            Exp(expC1W07, inC1W07);
            float32x4_t expC1Data00 = vld1q_f32(expC1W00);
            float32x4_t expC1Data01 = vld1q_f32(expC1W01);
            float32x4_t expC1Data02 = vld1q_f32(expC1W02);
            float32x4_t expC1Data03 = vld1q_f32(expC1W03);
            float32x4_t expC1Data04 = vld1q_f32(expC1W04);
            float32x4_t expC1Data05 = vld1q_f32(expC1W05);
            float32x4_t expC1Data06 = vld1q_f32(expC1W06);
            float32x4_t expC1Data07 = vld1q_f32(expC1W07);
            sumData00 = vaddq_f32(sumData00, expC1Data00);
            sumData01 = vaddq_f32(sumData01, expC1Data01);
            sumData02 = vaddq_f32(sumData02, expC1Data02);
            sumData03 = vaddq_f32(sumData03, expC1Data03);
            sumData04 = vaddq_f32(sumData04, expC1Data04);
            sumData05 = vaddq_f32(sumData05, expC1Data05);
            sumData06 = vaddq_f32(sumData06, expC1Data06);
            sumData07 = vaddq_f32(sumData07, expC1Data07);
            // store
            vst1q_f32(sumW00, sumData00);
            vst1q_f32(sumW01, sumData01);
            vst1q_f32(sumW02, sumData02);
            vst1q_f32(sumW03, sumData03);
            vst1q_f32(sumW04, sumData04);
            vst1q_f32(sumW05, sumData05);
            vst1q_f32(sumW06, sumData06);
            vst1q_f32(sumW07, sumData07);
#endif
        }
    }

    for (ci = 0; ci < channel; ci += 4) {
        float *expC = exp + ci * width;
        float *outC = dst + ci * width * height;
        for (size_t wi = 0; wi < width; ++wi) {
            size_t sw = wi * 4;
            float *sumW = sum + sw;
            float *expW = expC + sw;
            float *outW = outC + sw;
#if __ARM_NEON
            float32x4_t vecExp = vld1q_f32(expW);
            float32x4_t vecSum = vld1q_f32(sumW);
            float32_t sumVal = vaddvq_f32(vecSum);
            float32_t recipSum = 1 / sumVal;
            vst1q_f32(outW, vmulq_n_f32(vecExp, recipSum));
#else
            float sumVal = sumW[0] + sumW[1] + sumW[2] + sumW[3];
            for (size_t i = 0; i < 4; ++i) {
                outW[i] = expW[i] / sumVal;
            }
#endif // __ARM_NEON
        }
    }
}
#endif
