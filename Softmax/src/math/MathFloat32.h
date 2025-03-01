#include <KernelFunc.h>

#include <limits>
#include <stdio.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif


/******************************* Exponential Float Computing *******************************/

// Implementation in Assembly
static const float expParam[10] = {
    1.442695041f, 0.693147180f, 
    0.9999999916728642f, 1.000000059694879f, 
    0.5000006143673624f, 0.16666570253074878f, 
    0.04165989275009526f, 0.008336936973260111f, 
    0.0014122663401803872f, 0.00019578093328483123f
};

static inline void Exp(float *dst, const float *src)
{
    ExpKernel(dst, src, expParam);
    return;
}

// Implementation in NEON
#if __ARM_NEON
inline float32x4_t prefer_vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c)
{
#if __ARM_FEATURE_FMA
    return vfmaq_f32(a, b, c);
#else  // __ARM_FEATURE_FMA
    return vmlaq_f32(a, b, c);
#endif // __ARM_FEATURE_FMA
}

static const uint32_t exp_f32_coeff[] = {
    0x3f7ffff6, // x^1: 0x1.ffffecp-1f
    0x3efffedb, // x^2: 0x1.fffdb6p-2f
    0x3e2aaf33, // x^3: 0x1.555e66p-3f
    0x3d2b9f17, // x^4: 0x1.573e2ep-5f
    0x3c072010, // x^5: 0x1.0e4020p-7f
};

inline float32x4_t vexpq_f32(float32x4_t x)
{
    const auto c1 = vreinterpretq_f32_u32(vdupq_n_u32(exp_f32_coeff[0]));
    const auto c2 = vreinterpretq_f32_u32(vdupq_n_u32(exp_f32_coeff[1]));
    const auto c3 = vreinterpretq_f32_u32(vdupq_n_u32(exp_f32_coeff[2]));
    const auto c4 = vreinterpretq_f32_u32(vdupq_n_u32(exp_f32_coeff[3]));
    const auto c5 = vreinterpretq_f32_u32(vdupq_n_u32(exp_f32_coeff[4]));

    const auto shift   = vreinterpretq_f32_u32(vdupq_n_u32(0x4b00007f)); // 2^23 + 127 = 0x1.0000fep23f
    const auto inv_ln2 = vreinterpretq_f32_u32(vdupq_n_u32(0x3fb8aa3b)); // 1 / ln(2) = 0x1.715476p+0f
    const auto neg_ln2_hi = vreinterpretq_f32_u32(vdupq_n_u32(0xbf317200)); // -ln(2) from bits  -1 to -19: -0x1.62e400p-1f
    const auto neg_ln2_lo = vreinterpretq_f32_u32(vdupq_n_u32(0xb5bfbe8e)); // -ln(2) from bits -20 to -42: -0x1.7f7d1cp-20f

    const auto inf       = vdupq_n_f32(std::numeric_limits<float>::infinity());
    const auto max_input = vdupq_n_f32(88.37f); // Approximately ln(2^127.5)
    const auto zero      = vdupq_n_f32(0.f);
    const auto min_input = vdupq_n_f32(-86.64f); // Approximately ln(2^-125)

    // Range reduction:
    //   e^x = 2^n * e^r
    // where:
    //   n = floor(x / ln(2))
    //   r = x - n * ln(2)
    const auto z     = prefer_vfmaq_f32(shift, x, inv_ln2);
    const auto n     = z - shift;
    const auto scale = vreinterpretq_f32_u32(vreinterpretq_u32_f32(z) << 23); // 2^n

    const auto r_hi = prefer_vfmaq_f32(x, n, neg_ln2_hi);
    const auto r    = prefer_vfmaq_f32(r_hi, n, neg_ln2_lo);

    // e^r = scale * (1 + c1 * r + c2 * r^2 + c3 * r^3 + c4 * r^4 + c5 * r^5)
    const auto r2 = r * r;

    const auto p1     = c1 * r;
    const auto p23    = prefer_vfmaq_f32(c2, c3, r);
    const auto p45    = prefer_vfmaq_f32(c4, c5, r);
    const auto p2345  = prefer_vfmaq_f32(p23, p45, r2);
    const auto p12345 = prefer_vfmaq_f32(p1, p2345, r2);

    auto poly = prefer_vfmaq_f32(scale, p12345, scale);

    // Handle underflow and overflow.
    poly = vbslq_f32(vcltq_f32(x, min_input), zero, poly);
    poly = vbslq_f32(vcgtq_f32(x, max_input), inf, poly);

    return poly;
}

static inline void Exp(float *dst, float32x4_t src)
{
    vst1q_f32(dst, vexpq_f32(src));
    return;
}
#endif // __ARM_NEON
