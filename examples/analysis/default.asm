code_snippet:
    vpslldq $3, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vmulps %xmm0, %xmm0, %xmm1
    dec %rcx
    jne code_snippet
