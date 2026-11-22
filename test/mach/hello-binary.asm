
out/bin/example:     file format elf64-x86-64


Disassembly of section .text:

0000000000400000 <std.system.panic.panic_sink>:
panic_sink():
/app/dep/std/src/system/panic.mach:51
  400000:	55                                              	push   rbp

0000000000400040 <std.system.panic.panic>:
panic():
/app/dep/std/src/system/panic.mach:165
  400040:	55                                              	push   rbp

0000000000400080 <std.types.string.str_len>:
str_len():
/app/dep/std/src/types/string.mach:118
  400080:	48 83 ff 00                                     	cmp    rdi,0x0

0000000000400140 <std.system.os.linux.shared.syscall2>:
syscall2():
/app/dep/std/src/system/os/linux/shared.mach:639
  400140:	55                                              	push   rbp

0000000000400190 <std.system.os.linux.shared.syscall3>:
syscall3():
/app/dep/std/src/system/os/linux/shared.mach:678
  400190:	55                                              	push   rbp

00000000004001f0 <std.system.os.linux.shared.syscall4>:
syscall4():
/app/dep/std/src/system/os/linux/shared.mach:721
  4001f0:	55                                              	push   rbp

0000000000400250 <std.system.os.linux.shared.thread_block_prepare>:
thread_block_prepare():
/app/dep/std/src/system/os/linux/shared.mach:944
  400250:	48 89 3f                                        	mov    QWORD PTR [rdi],rdi

0000000000400270 <std.system.os.linux.shared.write>:
write():
/app/dep/std/src/system/os/linux/shared.mach:1237
  400270:	55                                              	push   rbp

0000000000400300 <std.system.os.linux.shared.pagesz_from_auxv>:
pagesz_from_auxv():
/app/dep/std/src/system/os/linux/shared.mach:1613
  400300:	55                                              	push   rbp

0000000000400320 <std.system.os.linux.shared.auxv_value>:
auxv_value():
/app/dep/std/src/system/os/linux/shared.mach:1624
  400320:	48 c7 c0 00 00 00 00                            	mov    rax,0x0

0000000000400390 <std.system.os.linux.shared.capture_pagesz>:
capture_pagesz():
/app/dep/std/src/system/os/linux/shared.mach:1642
  400390:	55                                              	push   rbp

00000000004003d0 <std.system.os.linux.shared.capture_main_stack>:
capture_main_stack():
/app/dep/std/src/system/os/linux/shared.mach:1664
  4003d0:	55                                              	push   rbp

0000000000400580 <std.system.os.linux.shared.loader_present>:
loader_present():
/app/dep/std/src/system/os/linux/shared.mach:1715
  400580:	55                                              	push   rbp

00000000004005d0 <std.system.os.linux.x86_64.thread_block_enter>:
thread_block_enter():
/app/dep/std/src/system/os/linux/x86_64.mach:251
  4005d0:	55                                              	push   rbp

0000000000400610 <std.runtime.linux.x86_64.install_main_block>:
install_main_block():
/app/dep/std/src/runtime/linux/x86_64.mach:52
  400610:	55                                              	push   rbp

0000000000400650 <_rt_init>:
_rt_init():
/app/dep/std/src/runtime/linux/x86_64.mach:60
  400650:	55                                              	push   rbp

0000000000400680 <_start>:
_start():
/app/dep/std/src/runtime/linux/x86_64.mach:79
  400680:	48 89 e0                                        	mov    rax,rsp

00000000004006e0 <std.system.os.errors.error_kind>:
error_kind():
/app/dep/std/src/system/os/errors.mach:26
  4006e0:	48 83 ff fc                                     	cmp    rdi,0xfffffffffffffffc

00000000004009a0 <std.system.os.native>:
native():
/app/dep/std/src/system/os.mach:65
  4009a0:	89 f8                                           	mov    eax,edi

00000000004009b0 <std.system.os.write>:
write():
/app/dep/std/src/system/os.mach:215
  4009b0:	55                                              	push   rbp

00000000004009f0 <std.system.os.error>:
error():
/app/dep/std/src/system/os.mach:539
  4009f0:	55                                              	push   rbp

0000000000400a40 <std.io.writer.native_failure>:
native_failure():
/app/dep/std/src/io/writer.mach:60
  400a40:	55                                              	push   rbp

0000000000400b10 <std.io.writer.persisted>:
persisted():
/app/dep/std/src/io/writer.mach:65
  400b10:	55                                              	push   rbp

0000000000400ba0 <std.io.writer.classify>:
classify():
/app/dep/std/src/io/writer.mach:73
  400ba0:	55                                              	push   rbp

0000000000400ce0 <std.io.writer.advance>:
advance():
/app/dep/std/src/io/writer.mach:81
  400ce0:	55                                              	push   rbp

0000000000400f70 <std.io.writer.write>:
write():
/app/dep/std/src/io/writer.mach:97
  400f70:	55                                              	push   rbp

00000000004013a0 <std.io.writer.write_all>:
write_all():
/app/dep/std/src/io/writer.mach:113
  4013a0:	55                                              	push   rbp

0000000000401660 <std.io.writer.reprefix>:
reprefix():
/app/dep/std/src/io/writer.mach:174
  401660:	55                                              	push   rbp

00000000004018b0 <std.io.writer.push>:
push():
/app/dep/std/src/io/writer.mach:182
  4018b0:	55                                              	push   rbp

0000000000401ad0 <std.io.writer.call_prefix>:
call_prefix():
/app/dep/std/src/io/writer.mach:199
  401ad0:	48 8b 47 28                                     	mov    rax,QWORD PTR [rdi+0x28]

0000000000401af0 <std.io.writer.buffered_write>:
buffered_write():
/app/dep/std/src/io/writer.mach:203
  401af0:	55                                              	push   rbp

00000000004022a0 <std.io.writer.buffered>:
buffered():
/app/dep/std/src/io/writer.mach:250
  4022a0:	48 83 ec 58                                     	sub    rsp,0x58

0000000000402370 <std.io.writer.flush>:
flush():
/app/dep/std/src/io/writer.mach:266
  402370:	55                                              	push   rbp

0000000000402770 <std.io.writer.pushed>:
pushed():
/app/dep/std/src/io/writer.mach:276
  402770:	48 8b 47 28                                     	mov    rax,QWORD PTR [rdi+0x28]

0000000000402780 <std.format.write_full>:
write_full():
/app/dep/std/src/format.mach:78
  402780:	55                                              	push   rbp

00000000004029c0 <std.format.write_str>:
write_str():
/app/dep/std/src/format.mach:99
  4029c0:	55                                              	push   rbp

0000000000402b00 <std.format.write_byte>:
write_byte():
/app/dep/std/src/format.mach:109
  402b00:	55                                              	push   rbp

0000000000402b80 <std.format.write_newline>:
write_newline():
/app/dep/std/src/format.mach:118
  402b80:	55                                              	push   rbp

0000000000402c00 <std.filesystem.write>:
write():
/app/dep/std/src/filesystem.mach:213
  402c00:	55                                              	push   rbp

0000000000402d70 <std.filesystem.file_write_cb>:
file_write_cb():
/app/dep/std/src/filesystem.mach:243
  402d70:	55                                              	push   rbp

0000000000402fd0 <std.filesystem.writer>:
writer():
/app/dep/std/src/filesystem.mach:265
  402fd0:	48 83 ec 38                                     	sub    rsp,0x38

0000000000403030 <std.print.stdout_writer>:
stdout_writer():
/app/dep/std/src/print.mach:36
  403030:	55                                              	push   rbp

0000000000403080 <std.print.settle_write>:
settle_write():
/app/dep/std/src/print.mach:66
  403080:	55                                              	push   rbp

0000000000403460 <std.print.staged_line>:
staged_line():
/app/dep/std/src/print.mach:76
  403460:	55                                              	push   rbp

0000000000403600 <std.print.write_line>:
write_line():
/app/dep/std/src/print.mach:100
  403600:	55                                              	push   rbp

00000000004039b0 <std.print.println>:
println():
/app/dep/std/src/print.mach:135
  4039b0:	55                                              	push   rbp

0000000000403a50 <main>:
main():
/app/src/example.mach:6
  403a50:	55                                              	push   rbp

