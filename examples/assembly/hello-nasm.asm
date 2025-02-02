; -------------------------------------------------------------------
; hello.asm
;
; Simple "Hello, world!" program for NASM 2.16.01 (32-bit)
;
; Assemble and link with:
;   nasm -f elf32 hello.asm -o hello.o
;   ld -m elf_i386 hello.o -o hello
; -------------------------------------------------------------------

global  _start

section .text
_start:
        ; Write "Hello, world!" to stdout (file descriptor 1)
        mov     eax, 4      ; sys_write
        mov     ebx, 1      ; stdout
        mov     ecx, msg    ; message address
        mov     edx, len    ; message length
        int     0x80        ; call kernel

        ; Exit program
        mov     eax, 1      ; sys_exit
        mov     ebx, 0      ; exit code 0
        int     0x80        ; call kernel

section .data
    msg:    db      "Hello, world!", 0xa
    len:    equ     $ - msg    ; Calculate message length
 