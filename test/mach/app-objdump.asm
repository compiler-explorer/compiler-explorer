
/app/out/obj/example/example.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <main>:
main():
/app/src/example.mach:9
   0:	55                                              	push   %rbp
   1:	48 89 e5                                        	mov    %rsp,%rbp
   4:	48 83 ec 50                                     	sub    $0x50,%rsp
   8:	48 89 5d b8                                     	mov    %rbx,-0x48(%rbp)
   c:	48 8d 5d c8                                     	lea    -0x38(%rbp),%rbx
  10:	48 8d 35 00 00 00 00                            	lea    0x0(%rip),%rsi        # 17 <main+0x17>
  17:	48 89 df                                        	mov    %rbx,%rdi
  1a:	e8 00 00 00 00                                  	call   1f <main+0x1f>
/app/src/example.mach:10
  1f:	e8 00 00 00 00                                  	call   24 <main+0x24>
  24:	48 8b 5d b8                                     	mov    -0x48(%rbp),%rbx
  28:	48 89 ec                                        	mov    %rbp,%rsp
  2b:	5d                                              	pop    %rbp
  2c:	c3                                              	ret
