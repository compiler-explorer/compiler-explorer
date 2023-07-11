    .intel_syntax noprefix

.data

.text
    .globl _f
_f:

    push rbp
    mov rbp, rsp
    sub rsp, 16
    jmp f_entry
f_entry:
    # a = $cmp neq x y
    mov r10, 0
    mov r11, 1
    cmp rdi, rsi
    cmovne r10, r11
    mov qword ptr [rbp + -8], r10
    cmp qword ptr [rbp + -8], 0
    jne f_t
    jmp f_f
f_f:
    mov rax, rsi
    jmp f_epilogue
f_t:
    mov rax, rdi
    jmp f_epilogue

f_epilogue:

    add rsp, 16
    pop rbp
    ret

