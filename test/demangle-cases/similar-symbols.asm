_Z2aai:
        push    rbp
        mov     rbp, rsp
        mov     DWORD PTR [rbp-4], edi
        nop
        pop     rbp
        ret
_Z2aaii:
        push    rbp
        mov     rbp, rsp
        mov     DWORD PTR [rbp-4], edi
        mov     DWORD PTR [rbp-8], esi
        nop
        pop     rbp
        ret
main:
        push    rbp
        mov     rbp, rsp
        mov     esi, 1
        mov     edi, 0
        call    _Z2aaii
        mov     edi, 1
        call    _Z2aai
        mov     eax, 0
        pop     rbp
        ret