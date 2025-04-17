_ZN7example29doesnt_work_with_mangled_name17hf9824bdcef81f607E:
        lea     rax, [rdi + 1]
        ret

works_with_no_mangling:
        lea     rax, [rdi + 2]
        ret

_ZN7example19non_public_function17hb49d2e80d343be8bE:
        lea     rax, [rdi + 3]
        ret

_ZN7example16another_function17h50be9330a2452582E:
        push    r15
        push    r14
        push    rbx
        mov     rbx, rdi
        call    qword ptr [rip + _ZN7example29doesnt_work_with_mangled_name17hf9824bdcef81f607E@GOTPCREL]
        mov     r14, rax
        mov     rdi, rbx
        call    qword ptr [rip + works_with_no_mangling@GOTPCREL]
        mov     r15, rax
        add     r15, r14
        mov     rdi, rbx
        call    _ZN7example19non_public_function17hb49d2e80d343be8bE
        add     rax, r15
        pop     rbx
        pop     r14
        pop     r15
        ret
