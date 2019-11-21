_Z1gv:
        subq    $8, %rsp
        call    *_ZN3foo3bazE+8(%rip)
        xorl    %eax, %eax
        addq    $8, %rsp
        ret