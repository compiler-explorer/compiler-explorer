_Z1gv:
        sub     rsp, 8
        call    [QWORD PTR _ZN3foo3bazE[rip+8]]
        xor     eax, eax
        add     rsp, 8
        ret