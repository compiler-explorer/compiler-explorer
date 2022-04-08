code_snippet:
    vpslldq xmm1, xmm0, 3
    vaddps xmm0, xmm0, xmm1
    vmulps xmm1, xmm0, xmm0
    dec rcx
    jne code_snippet
