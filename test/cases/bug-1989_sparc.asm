        .file   "example.d"
        .section        ".text"
        .align 4
        .weak   issue1989
        .type   issue1989, #function
        .proc   00
issue1989:
.LFB0:
        add     %o1, %o0, %o1
        mov     0, %g1
        cmp     %o1, %o0
        movg    %icc, 1, %g1
        jmp     %o7+8
         mov    %g1, %o0
.LFE0:
        .size   issue1989, .-issue1989
        .ident  "GCC: (GNU) 9.0.1 20190425 (experimental)"
        .section        .note.GNU-stack,"",@progbits
