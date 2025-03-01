#if defined(__aarch64__)

#include <include/assembly.h>
// void ExpKernel(
// auto:
// x0: float *dst,
// x1: const float *src,
// x2: const float *param)
ASM_FUNCTION ExpKernel

ld2r {v30.4s, v31.4s}, [x2], #8
ld2r {v0.4s, v1.4s}, [x2], #8
ld2r {v2.4s, v3.4s}, [x2], #8
ld2r {v4.4s, v5.4s}, [x2], #8
ld2r {v6.4s, v7.4s}, [x2]
movi v27.4s, #23
movi v29.4s, #87
scvtf v29.4s, v29.4s
fneg v28.4s, v29.4s

ld1 {v16.4s}, [x1]
fmin v16.4s, v16.4s, v29.4s
fmax v17.4s, v16.4s, v28.4s


fmul v16.4s, v17.4s, v30.4s
fcvtzs v16.4s, v16.4s
scvtf v20.4s, v16.4s
fmls v17.4s, v20.4s, v31.4s
fmul v18.4s, v17.4s, v17.4s
fmul v19.4s, v18.4s, v18.4s

fmla v0.4s, v1.4s, v17.4s
fmla v2.4s, v3.4s, v17.4s
fmla v4.4s, v5.4s, v17.4s
fmla v6.4s, v7.4s, v17.4s
fmla v0.4s, v2.4s, v18.4s
fmla v4.4s, v6.4s, v18.4s
fmla v0.4s, v4.4s, v19.4s

ushl v16.4s, v16.4s, v27.4s
add v0.4s, v0.4s, v16.4s

st1 {v0.4s}, [x0]

ret

#endif
