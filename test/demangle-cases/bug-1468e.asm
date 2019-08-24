g():                                  # @g()
        push    rax
        call    qword ptr [rip + _ZN3foo3bazE+8]
        xor     eax, eax
        pop     rcx
        ret