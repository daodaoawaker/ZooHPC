#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#ifdef __ELF__
.macro ASM_FUNCTION name
    .text
    .align 5
    .global \name
    .hidden \name
    .type \name, %function
    \name:
.endm
#endif

#endif // ASSEMBLY_H
