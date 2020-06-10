        .SECTION P,CODE
        .GLB    _square
_square:
        push.l  r10
        push.l  r6
        add     #-4, r0, r6
        mov.L   r6, r0
        mov.L   r1, [r6]
        mov.L   [r6], r10
        mul     r10, r10
        mov.L   r10, r1
        add     #4, r0
        pop     r6
        pop     r10
        rts
        .ident  "GCC: (GNU) 9.0.1 20190425 (experimental)"
