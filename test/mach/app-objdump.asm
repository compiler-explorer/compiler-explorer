
/app/out_bin/obj/example/example.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <main>:
main():
   0:	48 8d 64 24 a8       	lea    -0x58(%rsp),%rsp
/app/src/example.mach:5
   5:	48 89 3c 24          	mov    %rdi,(%rsp)
   9:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
/app/src/example.mach:6
   e:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  13:	48 8d 35 00 00 00 00 	lea    0x0(%rip),%rsi        # 1a <main+0x1a>
  1a:	e8 00 00 00 00       	call   1f <main+0x1f>
/app/src/example.mach:7
  1f:	31 c0                	xor    %eax,%eax
/app/src/example.mach:5
  21:	48 8d 64 24 58       	lea    0x58(%rsp),%rsp
  26:	c3                   	ret
