        .file   "example.cpp"
        .text
        .globl  main
        .type   main, @function
main:
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $16, %rbp
        movl    $5, -4(%rbp)
        movl    -4(%rbp), %edi
        call    "square(int)"
        movl    %eax, -8(%rbp)
        movl    -8(%rbp), %eax
        addq    $16, %rsp
        popq    %rbp
        ret
        .size   main, .-main

"square(int)":
        pushq   %rbp
        movq    %rsp, %rbp
        movl    %edi, -4(%rbp)
        movl    -4(%rbp), %eax
        imull   %eax, %eax
        popq    %rbp
        ret
        .size   "square(int)", .-"square(int)"