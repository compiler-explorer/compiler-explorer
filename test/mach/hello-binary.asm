
out/bin/example:     file format elf64-x86-64


Disassembly of section .text:

0000000000400000 <std.types.string.str_len>:
str_len():
/app/dep/std/src/types/string.mach:118
  400000:	48 83 ff 00                                     	cmp    rdi,0x0

00000000004000dd <std.system.os.linux.shared.syscall3>:
syscall3():
/app/dep/std/src/system/os/linux/shared.mach:668
  4000dd:	55                                              	push   rbp

000000000040013b <std.system.os.linux.shared.write>:
write():
/app/dep/std/src/system/os/linux/shared.mach:1186
  40013b:	55                                              	push   rbp

00000000004001dc <std.system.os.linux.shared.pagesz_from_auxv>:
pagesz_from_auxv():
/app/dep/std/src/system/os/linux/shared.mach:1564
  4001dc:	48 c7 c0 00 00 00 00                            	mov    rax,0x0

0000000000400256 <std.system.os.linux.shared.capture_pagesz>:
capture_pagesz():
/app/dep/std/src/system/os/linux/shared.mach:1582
  400256:	55                                              	push   rbp

0000000000400290 <_rt_init>:
_rt_init():
/app/dep/std/src/runtime/linux/x86_64.mach:41
  400290:	55                                              	push   rbp

00000000004002ac <_start>:
_start():
/app/dep/std/src/runtime/linux/x86_64.mach:56
  4002ac:	48 89 e0                                        	mov    rax,rsp

0000000000400300 <std.io.error.from_code>:
from_code():
/app/dep/std/src/io/error.mach:73
  400300:	55                                              	push   rbp

0000000000400580 <std.io.writer.native_failure>:
native_failure():
/app/dep/std/src/io/writer.mach:60
  400580:	55                                              	push   rbp

00000000004006a8 <std.io.writer.persisted>:
persisted():
/app/dep/std/src/io/writer.mach:65
  4006a8:	55                                              	push   rbp

000000000040073a <std.io.writer.classify>:
classify():
/app/dep/std/src/io/writer.mach:73
  40073a:	55                                              	push   rbp

00000000004008a3 <std.io.writer.advance>:
advance():
/app/dep/std/src/io/writer.mach:81
  4008a3:	55                                              	push   rbp

0000000000400bfa <std.io.writer.write>:
write():
/app/dep/std/src/io/writer.mach:97
  400bfa:	55                                              	push   rbp

00000000004010b4 <std.io.writer.write_all>:
write_all():
/app/dep/std/src/io/writer.mach:113
  4010b4:	55                                              	push   rbp

00000000004014b2 <std.io.writer.reprefix>:
reprefix():
/app/dep/std/src/io/writer.mach:174
  4014b2:	55                                              	push   rbp

0000000000401790 <std.io.writer.push>:
push():
/app/dep/std/src/io/writer.mach:182
  401790:	55                                              	push   rbp

0000000000401a70 <std.io.writer.call_prefix>:
call_prefix():
/app/dep/std/src/io/writer.mach:199
  401a70:	48 8b 47 28                                     	mov    rax,QWORD PTR [rdi+0x28]

0000000000401a8d <std.io.writer.buffered_write>:
buffered_write():
/app/dep/std/src/io/writer.mach:203
  401a8d:	55                                              	push   rbp

00000000004024af <std.io.writer.buffered>:
buffered():
/app/dep/std/src/io/writer.mach:250
  4024af:	55                                              	push   rbp

000000000040258f <std.io.writer.flush>:
flush():
/app/dep/std/src/io/writer.mach:266
  40258f:	55                                              	push   rbp

0000000000402a84 <std.io.writer.pushed>:
pushed():
/app/dep/std/src/io/writer.mach:276
  402a84:	48 8b 47 28                                     	mov    rax,QWORD PTR [rdi+0x28]

0000000000402a89 <std.io.writer.mem_write>:
mem_write():
/app/dep/std/src/io/writer.mach:292
  402a89:	55                                              	push   rbp

0000000000402e00 <std.filesystem.write>:
write():
/app/dep/std/src/filesystem.mach:197
  402e00:	55                                              	push   rbp

0000000000402f79 <std.filesystem.file_write_cb>:
file_write_cb():
/app/dep/std/src/filesystem.mach:227
  402f79:	55                                              	push   rbp

0000000000403240 <std.filesystem.writer>:
writer():
/app/dep/std/src/filesystem.mach:249
  403240:	55                                              	push   rbp

00000000004032a3 <std.format.write_full>:
write_full():
/app/dep/std/src/format.mach:71
  4032a3:	55                                              	push   rbp

0000000000403511 <std.format.write_str>:
write_str():
/app/dep/std/src/format.mach:92
  403511:	55                                              	push   rbp

0000000000403650 <std.format.write_byte>:
write_byte():
/app/dep/std/src/format.mach:102
  403650:	55                                              	push   rbp

00000000004036dc <std.format.write_newline>:
write_newline():
/app/dep/std/src/format.mach:111
  4036dc:	55                                              	push   rbp

0000000000403760 <std.print.stdout_writer>:
stdout_writer():
/app/dep/std/src/print.mach:29
  403760:	55                                              	push   rbp

00000000004037bb <std.print.settle_write>:
settle_write():
/app/dep/std/src/print.mach:59
  4037bb:	55                                              	push   rbp

0000000000403cd6 <std.print.staged_line>:
staged_line():
/app/dep/std/src/print.mach:69
  403cd6:	55                                              	push   rbp

0000000000403e76 <std.print.write_line>:
write_line():
/app/dep/std/src/print.mach:93
  403e76:	55                                              	push   rbp

00000000004042f0 <std.print.println>:
println():
/app/dep/std/src/print.mach:128
  4042f0:	55                                              	push   rbp

0000000000404390 <main>:
main():
/app/src/example.mach:6
  404390:	55                                              	push   rbp
