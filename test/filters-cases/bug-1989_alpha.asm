        .set noreorder
        .set volatile
        .set noat
        .set nomacro
        .arch ev4
        .text
        .align 2
        .align 4
        .weak   issue1989
        .ent issue1989
$issue1989..ng:
issue1989:
        .frame $30,0,$26,0
$LFB0:
        .cfi_startproc
        .prologue 0
        addl $16,$17,$0
        cmplt $16,$0,$0
        ret $31,($26),1
        .cfi_endproc
$LFE0:
        .end issue1989
        .ident  "GCC: (GNU) 9.0.1 20190425 (experimental)"
        .section        .note.GNU-stack,"",@progbits
