        .file   "example.cpp"
        .intel_syntax noprefix
        .text
.Ltext0:
        .globl  s_sA
        .section        .rodata
.LC0:
        .string "hello world!"
        .data
        .align 8
        .type   s_sA, @object
        .size   s_sA, 8
s_sA:
        .quad   .LC0
        .section        .rodata
        .align 8
        .type   s_sB, @object
        .size   s_sB, 10
s_sB:
        .string "hey there"
        .text
        .globl  main
        .type   main, @function
main:
.LFB0:
        .file 1 "/tmp/compiler-explorer-compiler116820-58-ewfj5u/example.cpp"
        .loc 1 6 0
        .cfi_startproc
        push    rbp
        .cfi_def_cfa_offset 16
        .cfi_offset 6, -16
        mov     rbp, rsp
        .cfi_def_cfa_register 6
        .loc 1 7 0
        mov     rax, QWORD PTR s_sA[rip]
        mov     rdi, rax
        call    puts
        .loc 1 8 0
        mov     edi, OFFSET FLAT:s_sB
        call    puts
        .loc 1 9 0
        mov     eax, 0
        pop     rbp
        .cfi_def_cfa 7, 8
        ret
        .cfi_endproc
