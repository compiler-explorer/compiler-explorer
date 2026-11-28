
/home/octalide/.gemini/antigravity-cli/brain/1455a340-1a46-4d4a-9f47-0d9f9dd1a57c/scratch/cap720/out_bin/bin/example:     file format elf64-x86-64

0000000000400160 <std.system.panic.panic_sink>:
  400160:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400165:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  40016a:	4c 8b 44 24 e8       	mov    -0x18(%rsp),%r8
00000000004001a0 <std.system.panic.panic>:
  4001a0:	48 8d 64 24 e8       	lea    -0x18(%rsp),%rsp
  4001a5:	48 89 3c 24          	mov    %rdi,(%rsp)
  4001a9:	31 c0                	xor    %eax,%eax
00000000004001e0 <std.crypto.ct.zeroize_plain>:
  4001e0:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  4001e5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  4001ea:	48 6b c6 01          	imul   $0x1,%rsi,%rax
0000000000400230 <std.types.string.str_len>:
  400230:	48 89 7c 24 c8       	mov    %rdi,-0x38(%rsp)
  400235:	48 89 f9             	mov    %rdi,%rcx
  400238:	48 85 c9             	test   %rcx,%rcx
0000000000400320 <std.system.os.linux.shared.syscall1>:
  400320:	48 89 7c 24 d8       	mov    %rdi,-0x28(%rsp)
  400325:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  40032a:	b8 00 00 00 00       	mov    $0x0,%eax
0000000000400370 <std.system.os.linux.shared.syscall2>:
  400370:	48 89 7c 24 d8       	mov    %rdi,-0x28(%rsp)
  400375:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  40037a:	48 89 54 24 e8       	mov    %rdx,-0x18(%rsp)
00000000004003d0 <std.system.os.linux.shared.syscall3>:
  4003d0:	48 89 5c 24 f8       	mov    %rbx,-0x8(%rsp)
  4003d5:	48 89 6c 24 f0       	mov    %rbp,-0x10(%rsp)
  4003da:	48 89 7c 24 b8       	mov    %rdi,-0x48(%rsp)
0000000000400450 <std.system.os.linux.shared.syscall6>:
  400450:	48 8d a4 24 38 ff ff 	lea    -0xc8(%rsp),%rsp
  400457:	ff 
  400458:	48 89 9c 24 c0 00 00 	mov    %rbx,0xc0(%rsp)
00000000004005e0 <std.system.os.linux.shared.allocate>:
  4005e0:	48 8d 64 24 d8       	lea    -0x28(%rsp),%rsp
  4005e5:	48 89 fa             	mov    %rdi,%rdx
  4005e8:	48 89 54 24 08       	mov    %rdx,0x8(%rsp)
0000000000400640 <std.system.os.linux.shared.thread_block_reserve>:
  400640:	b8 18 00 00 00       	mov    $0x18,%eax
  400645:	48 83 c0 0f          	add    $0xf,%rax
  400649:	b9 0f 00 00 00       	mov    $0xf,%ecx
0000000000400660 <std.system.os.linux.shared.thread_block_prepare>:
  400660:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400665:	48 89 f8             	mov    %rdi,%rax
  400668:	48 89 07             	mov    %rax,(%rdi)
0000000000400680 <std.system.os.linux.shared.capture_tls>:
  400680:	48 8d a4 24 68 ff ff 	lea    -0x98(%rsp),%rsp
  400687:	ff 
  400688:	48 89 9c 24 90 00 00 	mov    %rbx,0x90(%rsp)
00000000004008b0 <std.system.os.linux.shared.auxv_value>:
  4008b0:	48 89 7c 24 d8       	mov    %rdi,-0x28(%rsp)
  4008b5:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  4008ba:	31 c0                	xor    %eax,%eax
0000000000400900 <std.system.os.linux.shared.phdr_entry_size>:
  400900:	48 8d 64 24 e8       	lea    -0x18(%rsp),%rsp
  400905:	48 89 3c 24          	mov    %rdi,(%rsp)
  400909:	be 04 00 00 00       	mov    $0x4,%esi
0000000000400940 <std.system.os.linux.shared.phdr_field>:
  400940:	40 88 7c 24 e8       	mov    %dil,-0x18(%rsp)
  400945:	48 89 74 24 d8       	mov    %rsi,-0x28(%rsp)
  40094a:	48 89 54 24 e0       	mov    %rdx,-0x20(%rsp)
0000000000400960 <std.system.os.linux.shared.align_up>:
  400960:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400965:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  40096a:	48 01 f7             	add    %rsi,%rdi
0000000000400980 <std.system.os.linux.shared.tls_distance>:
  400980:	48 8d 64 24 e8       	lea    -0x18(%rsp),%rsp
  400985:	40 88 3c 24          	mov    %dil,(%rsp)
  400989:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
00000000004009e0 <std.system.os.linux.shared.thread_pointer_align>:
  4009e0:	48 8b 05 01 38 00 00 	mov    0x3801(%rip),%rax        # 4041e8 <std.system.os.linux.shared.tls_align>
  4009e7:	b9 10 00 00 00       	mov    $0x10,%ecx
  4009ec:	48 39 c1             	cmp    %rax,%rcx
0000000000400a00 <std.system.os.linux.shared.thread_area_size>:
  400a00:	48 8d 64 24 c8       	lea    -0x38(%rsp),%rsp
  400a05:	48 89 5c 24 30       	mov    %rbx,0x30(%rsp)
  400a0a:	48 89 6c 24 28       	mov    %rbp,0x28(%rsp)
0000000000400ab0 <std.system.os.linux.shared.thread_area_build>:
  400ab0:	48 8d 64 24 88       	lea    -0x78(%rsp),%rsp
  400ab5:	48 89 5c 24 70       	mov    %rbx,0x70(%rsp)
  400aba:	48 89 6c 24 68       	mov    %rbp,0x68(%rsp)
0000000000400bf0 <std.system.os.linux.shared.thread_tls_of>:
  400bf0:	48 8d 64 24 d8       	lea    -0x28(%rsp),%rsp
  400bf5:	48 89 5c 24 20       	mov    %rbx,0x20(%rsp)
  400bfa:	48 89 fb             	mov    %rdi,%rbx
0000000000400c60 <std.system.os.linux.shared.thread_block_at>:
  400c60:	48 8d 64 24 d8       	lea    -0x28(%rsp),%rsp
  400c65:	48 89 5c 24 20       	mov    %rbx,0x20(%rsp)
  400c6a:	48 89 fb             	mov    %rdi,%rbx
0000000000400cb0 <std.system.os.linux.shared.thread_pointer_of>:
  400cb0:	48 8d 64 24 d8       	lea    -0x28(%rsp),%rsp
  400cb5:	48 89 5c 24 20       	mov    %rbx,0x20(%rsp)
  400cba:	48 89 3c 24          	mov    %rdi,(%rsp)
0000000000400d00 <std.system.os.linux.shared.main_thread_block>:
  400d00:	48 8d 64 24 c8       	lea    -0x38(%rsp),%rsp
  400d05:	48 89 5c 24 30       	mov    %rbx,0x30(%rsp)
  400d0a:	48 89 6c 24 28       	mov    %rbp,0x28(%rsp)
0000000000400d90 <std.system.os.linux.shared.thread_tls_present>:
  400d90:	48 8b 05 69 34 00 00 	mov    0x3469(%rip),%rax        # 404200 <std.system.os.linux.shared.tls_memsz>
  400d97:	31 c9                	xor    %ecx,%ecx
  400d99:	48 85 c0             	test   %rax,%rax
0000000000400db0 <std.system.os.linux.shared.thread_tls_wipe>:
  400db0:	48 8d 64 24 d8       	lea    -0x28(%rsp),%rsp
  400db5:	48 89 3c 24          	mov    %rdi,(%rsp)
  400db9:	40 88 74 24 08       	mov    %sil,0x8(%rsp)
0000000000400e00 <std.system.os.linux.shared.write_vector>:
  400e00:	48 8d 64 24 b8       	lea    -0x48(%rsp),%rsp
  400e05:	48 89 5c 24 40       	mov    %rbx,0x40(%rsp)
  400e0a:	48 89 6c 24 38       	mov    %rbp,0x38(%rsp)
0000000000400e90 <std.system.os.linux.shared.auxv_of>:
  400e90:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400e95:	31 c0                	xor    %eax,%eax
  400e97:	48 89 44 24 f0       	mov    %rax,-0x10(%rsp)
0000000000400ed0 <std.system.os.linux.shared.capture_envp>:
  400ed0:	48 8d 64 24 e8       	lea    -0x18(%rsp),%rsp
  400ed5:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ed9:	48 89 3d 30 33 00 00 	mov    %rdi,0x3330(%rip)        # 404210 <std.system.os.linux.shared._envp>
0000000000400f10 <std.system.os.linux.shared.capture_pagesz>:
  400f10:	48 8d 64 24 f8       	lea    -0x8(%rsp),%rsp
  400f15:	48 8b 05 ec 32 00 00 	mov    0x32ec(%rip),%rax        # 404208 <std.system.os.linux.shared._auxv>
  400f1c:	48 85 c0             	test   %rax,%rax
0000000000400f50 <std.system.os.linux.shared.capture_thread_pointer>:
  400f50:	48 8d 64 24 f8       	lea    -0x8(%rsp),%rsp
  400f55:	e8 16 00 00 00       	call   400f70 <std.system.os.linux.shared.loader_present>
  400f5a:	31 c9                	xor    %ecx,%ecx
0000000000400f70 <std.system.os.linux.shared.loader_present>:
  400f70:	48 8d 64 24 f8       	lea    -0x8(%rsp),%rsp
  400f75:	48 8b 05 8c 32 00 00 	mov    0x328c(%rip),%rax        # 404208 <std.system.os.linux.shared._auxv>
  400f7c:	48 85 c0             	test   %rax,%rax
0000000000400fa0 <std.system.os.linux.shared.thread_pointer_owned>:
  400fa0:	0f b6 05 79 32 00 00 	movzbl 0x3279(%rip),%eax        # 404220 <std.system.os.linux.shared.thread_pointer_owned_>
  400fa7:	c3                   	ret
	...
0000000000400fb0 <std.system.os.linux.shared.loader_started>:
  400fb0:	48 8d 64 24 b8       	lea    -0x48(%rsp),%rsp
  400fb5:	48 89 5c 24 40       	mov    %rbx,0x40(%rsp)
  400fba:	48 89 6c 24 38       	mov    %rbp,0x38(%rsp)
0000000000401080 <std.system.os.linux.shared.terminate>:
  401080:	48 8d 64 24 e8       	lea    -0x18(%rsp),%rsp
  401085:	48 89 fe             	mov    %rdi,%rsi
  401088:	48 89 34 24          	mov    %rsi,(%rsp)
00000000004010a0 <std.system.os.linux.x86_64.thread_exit_wipe>:
  4010a0:	48 8d 64 24 f8       	lea    -0x8(%rsp),%rsp
  4010a5:	e8 e6 fc ff ff       	call   400d90 <std.system.os.linux.shared.thread_tls_present>
  4010aa:	3c 01                	cmp    $0x1,%al
00000000004010e0 <std.system.os.linux.x86_64.thread_block_enter>:
  4010e0:	48 8d 64 24 e8       	lea    -0x18(%rsp),%rsp
  4010e5:	48 89 5c 24 10       	mov    %rbx,0x10(%rsp)
  4010ea:	48 89 fb             	mov    %rdi,%rbx
0000000000401130 <std.system.os.linux.x86_64.thread_pointer>:
  401130:	b8 00 00 00 00       	mov    $0x0,%eax
  401135:	48 89 44 24 e8       	mov    %rax,-0x18(%rsp)
  40113a:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
0000000000401160 <std.system.os.linux.x86_64.main_thread_block>:
  401160:	48 8d 64 24 e8       	lea    -0x18(%rsp),%rsp
  401165:	48 89 3c 24          	mov    %rdi,(%rsp)
  401169:	0f b6 05 10 1e 00 00 	movzbl 0x1e10(%rip),%eax        # 402f80 <std.system.os.linux.x86_64.TLS_LAYOUT>
0000000000401190 <mach_thread_setup>:
  401190:	48 8d 64 24 e8       	lea    -0x18(%rsp),%rsp
  401195:	e8 e6 f4 ff ff       	call   400680 <std.system.os.linux.shared.capture_tls>
  40119a:	e8 b1 fd ff ff       	call   400f50 <std.system.os.linux.shared.capture_thread_pointer>
00000000004011f0 <_rt_init>:
  4011f0:	48 8d 64 24 f8       	lea    -0x8(%rsp),%rsp
  4011f5:	48 8b 3d 3c 30 00 00 	mov    0x303c(%rip),%rdi        # 404238 <_rt_envp>
  4011fc:	e8 cf fc ff ff       	call   400ed0 <std.system.os.linux.shared.capture_envp>
0000000000401220 <_rt_exit>:
  401220:	48 8d 64 24 e8       	lea    -0x18(%rsp),%rsp
  401225:	48 89 5c 24 10       	mov    %rbx,0x10(%rsp)
  40122a:	48 89 fb             	mov    %rdi,%rbx
0000000000401240 <_start>:
  401240:	48 89 e0             	mov    %rsp,%rax
  401243:	48 8b 38             	mov    (%rax),%rdi
  401246:	48 8d 70 08          	lea    0x8(%rax),%rsi
0000000000401290 <std.system.os.errors.error_kind>:
  401290:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  401295:	31 c0                	xor    %eax,%eax
  401297:	48 89 44 24 f0       	mov    %rax,-0x10(%rsp)
00000000004012f0 <std.system.os.native>:
  4012f0:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  4012f5:	48 89 f8             	mov    %rdi,%rax
  4012f8:	c3                   	ret
0000000000401300 <std.system.os.write_vector>:
  401300:	48 8d 64 24 c8       	lea    -0x38(%rsp),%rsp
  401305:	48 89 5c 24 30       	mov    %rbx,0x30(%rsp)
  40130a:	48 89 6c 24 28       	mov    %rbp,0x28(%rsp)
0000000000401350 <std.system.os.error>:
  401350:	48 8d 64 24 d8       	lea    -0x28(%rsp),%rsp
  401355:	48 89 5c 24 20       	mov    %rbx,0x20(%rsp)
  40135a:	48 89 6c 24 18       	mov    %rbp,0x18(%rsp)
00000000004013b0 <std.io.writer.native_failure>:
  4013b0:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  4013b5:	48 89 74 24 c8       	mov    %rsi,-0x38(%rsp)
  4013ba:	48 8b 08             	mov    (%rax),%rcx
0000000000401410 <std.io.writer.classify>:
  401410:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  401415:	48 8b 08             	mov    (%rax),%rcx
  401418:	48 8b 50 08          	mov    0x8(%rax),%rdx
00000000004014b0 <std.filesystem.write_vector>:
  4014b0:	48 8d 64 24 d8       	lea    -0x28(%rsp),%rsp
  4014b5:	48 89 5c 24 20       	mov    %rbx,0x20(%rsp)
  4014ba:	48 89 fb             	mov    %rdi,%rbx
0000000000401510 <std.print.write_pieces>:
  401510:	48 8d a4 24 18 ff ff 	lea    -0xe8(%rsp),%rsp
  401517:	ff 
  401518:	48 89 9c 24 e0 00 00 	mov    %rbx,0xe0(%rsp)
0000000000401750 <std.print.write_text>:
  401750:	48 8d a4 24 78 ff ff 	lea    -0x88(%rsp),%rsp
  401757:	ff 
  401758:	48 89 9c 24 80 00 00 	mov    %rbx,0x80(%rsp)
0000000000401830 <std.print.println>:
  401830:	48 8d 64 24 e8       	lea    -0x18(%rsp),%rsp
  401835:	48 89 5c 24 10       	mov    %rbx,0x10(%rsp)
  40183a:	48 89 fb             	mov    %rdi,%rbx
0000000000401870 <main>:
  401870:	48 8d 64 24 a8       	lea    -0x58(%rsp),%rsp
  401875:	48 89 3c 24          	mov    %rdi,(%rsp)
  401879:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
