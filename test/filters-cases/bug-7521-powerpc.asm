_ZN14some_namespace8functionEv:
        .quad   .Lfunc_begin0
        .quad   .TOC.@tocbase
        .quad   0
.Lfunc_begin0:
        blr

main:
        .quad   .Lfunc_begin1
        .quad   .TOC.@tocbase
        .quad   0
.Lfunc_begin1:
        mflr 0
        std 31, -8(1)
        stdu 1, -144(1)
        std 0, 160(1)
        mr      31, 1
        stw 3, 132(31)
        std 4, 120(31)
        addis 3, 2, .L.str@toc@ha
        addi 3, 3, .L.str@toc@l
        addis 4, 2, _ZN14some_namespace8functionEv@toc@ha
        addi 4, 4, _ZN14some_namespace8functionEv@toc@l
        bl _ZSt6printfPKcz
        nop
        li 3, 0
        addi 1, 1, 144
        ld 0, 16(1)
        ld 31, -8(1)
        mtlr 0
        blr

.L.str:
        .asciz  "%zu\n"
