
a.out:     file format elf64-x86-64


Disassembly of section .init:

0000000000401000 <_init>:
  401000:	f3 0f 1e fa          	endbr64
  401004:	48 83 ec 08          	sub    $0x8,%rsp
  401008:	48 c7 c0 00 00 00 00 	mov    $0x0,%rax
  40100f:	48 85 c0             	test   %rax,%rax
  401012:	74 02                	je     401016 <_init+0x16>
  401014:	ff d0                	call   *%rax
  401016:	48 83 c4 08          	add    $0x8,%rsp
  40101a:	c3                   	ret

Disassembly of section .plt:

0000000000401020 <.plt>:
  401020:	f3 0f 1e fa          	endbr64
  401024:	ff 25 d6 ff 0a 00    	jmp    *0xaffd6(%rip)        # 4b1000 <_GLOBAL_OFFSET_TABLE_+0x18>
  40102a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401030:	f3 0f 1e fa          	endbr64
  401034:	ff 25 ce ff 0a 00    	jmp    *0xaffce(%rip)        # 4b1008 <_GLOBAL_OFFSET_TABLE_+0x20>
  40103a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401040:	f3 0f 1e fa          	endbr64
  401044:	ff 25 c6 ff 0a 00    	jmp    *0xaffc6(%rip)        # 4b1010 <_GLOBAL_OFFSET_TABLE_+0x28>
  40104a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401050:	f3 0f 1e fa          	endbr64
  401054:	ff 25 be ff 0a 00    	jmp    *0xaffbe(%rip)        # 4b1018 <_GLOBAL_OFFSET_TABLE_+0x30>
  40105a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401060:	f3 0f 1e fa          	endbr64
  401064:	ff 25 b6 ff 0a 00    	jmp    *0xaffb6(%rip)        # 4b1020 <_GLOBAL_OFFSET_TABLE_+0x38>
  40106a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401070:	f3 0f 1e fa          	endbr64
  401074:	ff 25 ae ff 0a 00    	jmp    *0xaffae(%rip)        # 4b1028 <_GLOBAL_OFFSET_TABLE_+0x40>
  40107a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401080:	f3 0f 1e fa          	endbr64
  401084:	ff 25 a6 ff 0a 00    	jmp    *0xaffa6(%rip)        # 4b1030 <_GLOBAL_OFFSET_TABLE_+0x48>
  40108a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401090:	f3 0f 1e fa          	endbr64
  401094:	ff 25 9e ff 0a 00    	jmp    *0xaff9e(%rip)        # 4b1038 <_GLOBAL_OFFSET_TABLE_+0x50>
  40109a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4010a0:	f3 0f 1e fa          	endbr64
  4010a4:	ff 25 96 ff 0a 00    	jmp    *0xaff96(%rip)        # 4b1040 <_GLOBAL_OFFSET_TABLE_+0x58>
  4010aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4010b0:	f3 0f 1e fa          	endbr64
  4010b4:	ff 25 8e ff 0a 00    	jmp    *0xaff8e(%rip)        # 4b1048 <_GLOBAL_OFFSET_TABLE_+0x60>
  4010ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4010c0:	f3 0f 1e fa          	endbr64
  4010c4:	ff 25 86 ff 0a 00    	jmp    *0xaff86(%rip)        # 4b1050 <_GLOBAL_OFFSET_TABLE_+0x68>
  4010ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4010d0:	f3 0f 1e fa          	endbr64
  4010d4:	ff 25 7e ff 0a 00    	jmp    *0xaff7e(%rip)        # 4b1058 <_GLOBAL_OFFSET_TABLE_+0x70>
  4010da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4010e0:	f3 0f 1e fa          	endbr64
  4010e4:	ff 25 76 ff 0a 00    	jmp    *0xaff76(%rip)        # 4b1060 <_GLOBAL_OFFSET_TABLE_+0x78>
  4010ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4010f0:	f3 0f 1e fa          	endbr64
  4010f4:	ff 25 6e ff 0a 00    	jmp    *0xaff6e(%rip)        # 4b1068 <_GLOBAL_OFFSET_TABLE_+0x80>
  4010fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401100:	f3 0f 1e fa          	endbr64
  401104:	ff 25 66 ff 0a 00    	jmp    *0xaff66(%rip)        # 4b1070 <_GLOBAL_OFFSET_TABLE_+0x88>
  40110a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401110:	f3 0f 1e fa          	endbr64
  401114:	ff 25 5e ff 0a 00    	jmp    *0xaff5e(%rip)        # 4b1078 <_GLOBAL_OFFSET_TABLE_+0x90>
  40111a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401120:	f3 0f 1e fa          	endbr64
  401124:	ff 25 56 ff 0a 00    	jmp    *0xaff56(%rip)        # 4b1080 <_GLOBAL_OFFSET_TABLE_+0x98>
  40112a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401130:	f3 0f 1e fa          	endbr64
  401134:	ff 25 4e ff 0a 00    	jmp    *0xaff4e(%rip)        # 4b1088 <_GLOBAL_OFFSET_TABLE_+0xa0>
  40113a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401140:	f3 0f 1e fa          	endbr64
  401144:	ff 25 46 ff 0a 00    	jmp    *0xaff46(%rip)        # 4b1090 <_GLOBAL_OFFSET_TABLE_+0xa8>
  40114a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401150:	f3 0f 1e fa          	endbr64
  401154:	ff 25 3e ff 0a 00    	jmp    *0xaff3e(%rip)        # 4b1098 <_GLOBAL_OFFSET_TABLE_+0xb0>
  40115a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401160:	f3 0f 1e fa          	endbr64
  401164:	ff 25 36 ff 0a 00    	jmp    *0xaff36(%rip)        # 4b10a0 <_GLOBAL_OFFSET_TABLE_+0xb8>
  40116a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401170:	f3 0f 1e fa          	endbr64
  401174:	ff 25 2e ff 0a 00    	jmp    *0xaff2e(%rip)        # 4b10a8 <_GLOBAL_OFFSET_TABLE_+0xc0>
  40117a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

Disassembly of section .text:

0000000000401180 <__libc_message_impl.cold>:
  401180:	e8 0d 00 00 00       	call   401192 <abort>

0000000000401185 <_dl_start>:
  401185:	f3 0f 1e fa          	endbr64
  401189:	55                   	push   %rbp
  40118a:	48 89 e5             	mov    %rsp,%rbp
  40118d:	e8 00 00 00 00       	call   401192 <abort>

0000000000401192 <abort>:
  401192:	f3 0f 1e fa          	endbr64
  401196:	55                   	push   %rbp
  401197:	48 89 e5             	mov    %rsp,%rbp
  40119a:	53                   	push   %rbx
  40119b:	bb 0e 00 00 00       	mov    $0xe,%ebx
  4011a0:	48 81 ec b8 00 00 00 	sub    $0xb8,%rsp
  4011a7:	64 48 8b 3c 25 28 00 	mov    %fs:0x28,%rdi
  4011ae:	00 00 
  4011b0:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4011b4:	bf 06 00 00 00       	mov    $0x6,%edi
  4011b9:	e8 82 76 05 00       	call   458840 <raise>
  4011be:	41 ba 08 00 00 00    	mov    $0x8,%r10d
  4011c4:	31 d2                	xor    %edx,%edx
  4011c6:	31 ff                	xor    %edi,%edi
  4011c8:	48 8d 35 89 4e 08 00 	lea    0x84e89(%rip),%rsi        # 486058 <sigall_set>
  4011cf:	89 d8                	mov    %ebx,%eax
  4011d1:	0f 05                	syscall
  4011d3:	48 8d 3d c6 6a 0b 00 	lea    0xb6ac6(%rip),%rdi        # 4b7ca0 <lock>
  4011da:	e8 51 cb 02 00       	call   42dd30 <___pthread_rwlock_wrlock>
  4011df:	31 c0                	xor    %eax,%eax
  4011e1:	48 8d bd 50 ff ff ff 	lea    -0xb0(%rbp),%rdi
  4011e8:	31 d2                	xor    %edx,%edx
  4011ea:	b9 26 00 00 00       	mov    $0x26,%ecx
  4011ef:	48 8d b5 50 ff ff ff 	lea    -0xb0(%rbp),%rsi
  4011f6:	f3 ab                	rep stos %eax,%es:(%rdi)
  4011f8:	bf 06 00 00 00       	mov    $0x6,%edi
  4011fd:	48 c7 85 58 ff ff ff 	movq   $0xffffffffffffffff,-0xa8(%rbp)
  401204:	ff ff ff ff 
  401208:	e8 a3 76 05 00       	call   4588b0 <__libc_sigaction>
  40120d:	bf 06 00 00 00       	mov    $0x6,%edi
  401212:	e8 09 b3 02 00       	call   42c520 <__pthread_raise_internal>
  401217:	48 8d b5 48 ff ff ff 	lea    -0xb8(%rbp),%rsi
  40121e:	31 d2                	xor    %edx,%edx
  401220:	89 d8                	mov    %ebx,%eax
  401222:	48 c7 85 48 ff ff ff 	movq   $0x20,-0xb8(%rbp)
  401229:	20 00 00 00 
  40122d:	41 ba 08 00 00 00    	mov    $0x8,%r10d
  401233:	bf 01 00 00 00       	mov    $0x1,%edi
  401238:	0f 05                	syscall
  40123a:	f4                   	hlt
  40123b:	bf 7f 00 00 00       	mov    $0x7f,%edi
  401240:	e8 ab 12 01 00       	call   4124f0 <_exit>

0000000000401245 <_IO_fputs.cold>:
  401245:	f6 43 01 80          	testb  $0x80,0x1(%rbx)
  401249:	75 21                	jne    40126c <_IO_fputs.cold+0x27>
  40124b:	48 8b bb 88 00 00 00 	mov    0x88(%rbx),%rdi
  401252:	80 3d 1f 08 0b 00 00 	cmpb   $0x0,0xb081f(%rip)        # 4b1a78 <__libc_single_threaded>
  401259:	8b 47 04             	mov    0x4(%rdi),%eax
  40125c:	74 16                	je     401274 <_IO_fputs.cold+0x2f>
  40125e:	85 c0                	test   %eax,%eax
  401260:	75 2a                	jne    40128c <_IO_fputs.cold+0x47>
  401262:	31 c9                	xor    %ecx,%ecx
  401264:	31 f6                	xor    %esi,%esi
  401266:	48 89 4f 08          	mov    %rcx,0x8(%rdi)
  40126a:	89 37                	mov    %esi,(%rdi)
  40126c:	4c 89 e7             	mov    %r12,%rdi
  40126f:	e8 ec e6 07 00       	call   47f960 <_Unwind_Resume>
  401274:	85 c0                	test   %eax,%eax
  401276:	75 14                	jne    40128c <_IO_fputs.cold+0x47>
  401278:	31 d2                	xor    %edx,%edx
  40127a:	48 89 57 08          	mov    %rdx,0x8(%rdi)
  40127e:	87 07                	xchg   %eax,(%rdi)
  401280:	83 e8 01             	sub    $0x1,%eax
  401283:	7e e7                	jle    40126c <_IO_fputs.cold+0x27>
  401285:	e8 f6 45 00 00       	call   405880 <__lll_lock_wake_private>
  40128a:	eb e0                	jmp    40126c <_IO_fputs.cold+0x27>
  40128c:	83 e8 01             	sub    $0x1,%eax
  40128f:	89 47 04             	mov    %eax,0x4(%rdi)
  401292:	eb d8                	jmp    40126c <_IO_fputs.cold+0x27>

0000000000401294 <_IO_fwrite.cold>:
  401294:	f6 43 01 80          	testb  $0x80,0x1(%rbx)
  401298:	75 21                	jne    4012bb <_IO_fwrite.cold+0x27>
  40129a:	48 8b bb 88 00 00 00 	mov    0x88(%rbx),%rdi
  4012a1:	80 3d d0 07 0b 00 00 	cmpb   $0x0,0xb07d0(%rip)        # 4b1a78 <__libc_single_threaded>
  4012a8:	8b 47 04             	mov    0x4(%rdi),%eax
  4012ab:	74 16                	je     4012c3 <_IO_fwrite.cold+0x2f>
  4012ad:	85 c0                	test   %eax,%eax
  4012af:	75 2a                	jne    4012db <_IO_fwrite.cold+0x47>
  4012b1:	31 c9                	xor    %ecx,%ecx
  4012b3:	31 f6                	xor    %esi,%esi
  4012b5:	48 89 4f 08          	mov    %rcx,0x8(%rdi)
  4012b9:	89 37                	mov    %esi,(%rdi)
  4012bb:	4c 89 e7             	mov    %r12,%rdi
  4012be:	e8 9d e6 07 00       	call   47f960 <_Unwind_Resume>
  4012c3:	85 c0                	test   %eax,%eax
  4012c5:	75 14                	jne    4012db <_IO_fwrite.cold+0x47>
  4012c7:	31 d2                	xor    %edx,%edx
  4012c9:	48 89 57 08          	mov    %rdx,0x8(%rdi)
  4012cd:	87 07                	xchg   %eax,(%rdi)
  4012cf:	83 e8 01             	sub    $0x1,%eax
  4012d2:	7e e7                	jle    4012bb <_IO_fwrite.cold+0x27>
  4012d4:	e8 a7 45 00 00       	call   405880 <__lll_lock_wake_private>
  4012d9:	eb e0                	jmp    4012bb <_IO_fwrite.cold+0x27>
  4012db:	83 e8 01             	sub    $0x1,%eax
  4012de:	89 47 04             	mov    %eax,0x4(%rdi)
  4012e1:	eb d8                	jmp    4012bb <_IO_fwrite.cold+0x27>

00000000004012e3 <_IO_new_file_underflow.cold>:
  4012e3:	41 f6 44 24 01 80    	testb  $0x80,0x1(%r12)
  4012e9:	75 22                	jne    40130d <_IO_new_file_underflow.cold+0x2a>
  4012eb:	49 8b bc 24 88 00 00 	mov    0x88(%r12),%rdi
  4012f2:	00 
  4012f3:	80 3d 7e 07 0b 00 00 	cmpb   $0x0,0xb077e(%rip)        # 4b1a78 <__libc_single_threaded>
  4012fa:	8b 47 04             	mov    0x4(%rdi),%eax
  4012fd:	74 16                	je     401315 <_IO_new_file_underflow.cold+0x32>
  4012ff:	85 c0                	test   %eax,%eax
  401301:	75 2a                	jne    40132d <_IO_new_file_underflow.cold+0x4a>
  401303:	31 c9                	xor    %ecx,%ecx
  401305:	31 f6                	xor    %esi,%esi
  401307:	48 89 4f 08          	mov    %rcx,0x8(%rdi)
  40130b:	89 37                	mov    %esi,(%rdi)
  40130d:	48 89 df             	mov    %rbx,%rdi
  401310:	e8 4b e6 07 00       	call   47f960 <_Unwind_Resume>
  401315:	85 c0                	test   %eax,%eax
  401317:	75 14                	jne    40132d <_IO_new_file_underflow.cold+0x4a>
  401319:	31 d2                	xor    %edx,%edx
  40131b:	48 89 57 08          	mov    %rdx,0x8(%rdi)
  40131f:	87 07                	xchg   %eax,(%rdi)
  401321:	83 e8 01             	sub    $0x1,%eax
  401324:	7e e7                	jle    40130d <_IO_new_file_underflow.cold+0x2a>
  401326:	e8 55 45 00 00       	call   405880 <__lll_lock_wake_private>
  40132b:	eb e0                	jmp    40130d <_IO_new_file_underflow.cold+0x2a>
  40132d:	83 e8 01             	sub    $0x1,%eax
  401330:	89 47 04             	mov    %eax,0x4(%rdi)
  401333:	eb d8                	jmp    40130d <_IO_new_file_underflow.cold+0x2a>

0000000000401335 <_nl_load_domain.cold>:
  401335:	e8 58 fe ff ff       	call   401192 <abort>

000000000040133a <__printf_fp_buffer_1.isra.0.cold>:
  40133a:	e8 53 fe ff ff       	call   401192 <abort>

000000000040133f <__printf_fphex_buffer.cold>:
  40133f:	e8 4e fe ff ff       	call   401192 <abort>

0000000000401344 <_IO_new_fclose.cold>:
  401344:	f6 43 01 80          	testb  $0x80,0x1(%rbx)
  401348:	75 21                	jne    40136b <_IO_new_fclose.cold+0x27>
  40134a:	48 8b bb 88 00 00 00 	mov    0x88(%rbx),%rdi
  401351:	80 3d 20 07 0b 00 00 	cmpb   $0x0,0xb0720(%rip)        # 4b1a78 <__libc_single_threaded>
  401358:	8b 47 04             	mov    0x4(%rdi),%eax
  40135b:	74 16                	je     401373 <_IO_new_fclose.cold+0x2f>
  40135d:	85 c0                	test   %eax,%eax
  40135f:	75 2a                	jne    40138b <_IO_new_fclose.cold+0x47>
  401361:	31 c9                	xor    %ecx,%ecx
  401363:	31 f6                	xor    %esi,%esi
  401365:	48 89 4f 08          	mov    %rcx,0x8(%rdi)
  401369:	89 37                	mov    %esi,(%rdi)
  40136b:	4c 89 e7             	mov    %r12,%rdi
  40136e:	e8 ed e5 07 00       	call   47f960 <_Unwind_Resume>
  401373:	85 c0                	test   %eax,%eax
  401375:	75 14                	jne    40138b <_IO_new_fclose.cold+0x47>
  401377:	31 d2                	xor    %edx,%edx
  401379:	48 89 57 08          	mov    %rdx,0x8(%rdi)
  40137d:	87 07                	xchg   %eax,(%rdi)
  40137f:	83 e8 01             	sub    $0x1,%eax
  401382:	7e e7                	jle    40136b <_IO_new_fclose.cold+0x27>
  401384:	e8 f7 44 00 00       	call   405880 <__lll_lock_wake_private>
  401389:	eb e0                	jmp    40136b <_IO_new_fclose.cold+0x27>
  40138b:	83 e8 01             	sub    $0x1,%eax
  40138e:	89 47 04             	mov    %eax,0x4(%rdi)
  401391:	eb d8                	jmp    40136b <_IO_new_fclose.cold+0x27>

0000000000401393 <__getdelim.cold>:
  401393:	f6 43 01 80          	testb  $0x80,0x1(%rbx)
  401397:	75 21                	jne    4013ba <__getdelim.cold+0x27>
  401399:	48 8b bb 88 00 00 00 	mov    0x88(%rbx),%rdi
  4013a0:	80 3d d1 06 0b 00 00 	cmpb   $0x0,0xb06d1(%rip)        # 4b1a78 <__libc_single_threaded>
  4013a7:	8b 47 04             	mov    0x4(%rdi),%eax
  4013aa:	74 16                	je     4013c2 <__getdelim.cold+0x2f>
  4013ac:	85 c0                	test   %eax,%eax
  4013ae:	75 2a                	jne    4013da <__getdelim.cold+0x47>
  4013b0:	31 c9                	xor    %ecx,%ecx
  4013b2:	31 f6                	xor    %esi,%esi
  4013b4:	48 89 4f 08          	mov    %rcx,0x8(%rdi)
  4013b8:	89 37                	mov    %esi,(%rdi)
  4013ba:	4c 89 e7             	mov    %r12,%rdi
  4013bd:	e8 9e e5 07 00       	call   47f960 <_Unwind_Resume>
  4013c2:	85 c0                	test   %eax,%eax
  4013c4:	75 14                	jne    4013da <__getdelim.cold+0x47>
  4013c6:	31 d2                	xor    %edx,%edx
  4013c8:	48 89 57 08          	mov    %rdx,0x8(%rdi)
  4013cc:	87 07                	xchg   %eax,(%rdi)
  4013ce:	83 e8 01             	sub    $0x1,%eax
  4013d1:	7e e7                	jle    4013ba <__getdelim.cold+0x27>
  4013d3:	e8 a8 44 00 00       	call   405880 <__lll_lock_wake_private>
  4013d8:	eb e0                	jmp    4013ba <__getdelim.cold+0x27>
  4013da:	83 e8 01             	sub    $0x1,%eax
  4013dd:	89 47 04             	mov    %eax,0x4(%rdi)
  4013e0:	eb d8                	jmp    4013ba <__getdelim.cold+0x27>

00000000004013e2 <_IO_wfile_underflow.cold>:
  4013e2:	41 f6 44 24 01 80    	testb  $0x80,0x1(%r12)
  4013e8:	75 25                	jne    40140f <_IO_wfile_underflow.cold+0x2d>
  4013ea:	49 8b bc 24 88 00 00 	mov    0x88(%r12),%rdi
  4013f1:	00 
  4013f2:	80 3d 7f 06 0b 00 00 	cmpb   $0x0,0xb067f(%rip)        # 4b1a78 <__libc_single_threaded>
  4013f9:	8b 47 04             	mov    0x4(%rdi),%eax
  4013fc:	74 28                	je     401426 <_IO_wfile_underflow.cold+0x44>
  4013fe:	85 c0                	test   %eax,%eax
  401400:	75 3c                	jne    40143e <_IO_wfile_underflow.cold+0x5c>
  401402:	45 31 c0             	xor    %r8d,%r8d
  401405:	45 31 c9             	xor    %r9d,%r9d
  401408:	4c 89 47 08          	mov    %r8,0x8(%rdi)
  40140c:	44 89 0f             	mov    %r9d,(%rdi)
  40140f:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  401413:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
  40141a:	00 00 
  40141c:	75 28                	jne    401446 <_IO_wfile_underflow.cold+0x64>
  40141e:	48 89 df             	mov    %rbx,%rdi
  401421:	e8 3a e5 07 00       	call   47f960 <_Unwind_Resume>
  401426:	85 c0                	test   %eax,%eax
  401428:	75 14                	jne    40143e <_IO_wfile_underflow.cold+0x5c>
  40142a:	31 f6                	xor    %esi,%esi
  40142c:	48 89 77 08          	mov    %rsi,0x8(%rdi)
  401430:	87 07                	xchg   %eax,(%rdi)
  401432:	83 e8 01             	sub    $0x1,%eax
  401435:	7e d8                	jle    40140f <_IO_wfile_underflow.cold+0x2d>
  401437:	e8 44 44 00 00       	call   405880 <__lll_lock_wake_private>
  40143c:	eb d1                	jmp    40140f <_IO_wfile_underflow.cold+0x2d>
  40143e:	83 e8 01             	sub    $0x1,%eax
  401441:	89 47 04             	mov    %eax,0x4(%rdi)
  401444:	eb c9                	jmp    40140f <_IO_wfile_underflow.cold+0x2d>
  401446:	e8 55 21 01 00       	call   4135a0 <__stack_chk_fail>

000000000040144b <__nptl_free_stacks.cold>:
  40144b:	e8 42 fd ff ff       	call   401192 <abort>

0000000000401450 <__pthread_once_slow.isra.0.cold>:
  401450:	83 7d b0 00          	cmpl   $0x0,-0x50(%rbp)
  401454:	74 16                	je     40146c <__pthread_once_slow.isra.0.cold+0x1c>
  401456:	48 8b 7d a8          	mov    -0x58(%rbp),%rdi
  40145a:	ff 55 a0             	call   *-0x60(%rbp)
  40145d:	31 c0                	xor    %eax,%eax
  40145f:	31 f6                	xor    %esi,%esi
  401461:	4c 89 ef             	mov    %r13,%rdi
  401464:	89 45 b0             	mov    %eax,-0x50(%rbp)
  401467:	e8 64 0d 07 00       	call   4721d0 <__pthread_cleanup_pop>
  40146c:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  401470:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
  401477:	00 00 
  401479:	75 08                	jne    401483 <__pthread_once_slow.isra.0.cold+0x33>
  40147b:	48 89 df             	mov    %rbx,%rdi
  40147e:	e8 dd e4 07 00       	call   47f960 <_Unwind_Resume>
  401483:	e8 18 21 01 00       	call   4135a0 <__stack_chk_fail>

0000000000401488 <__printf_buffer_flush.cold>:
  401488:	0f 0b                	ud2

000000000040148a <__wprintf_buffer_flush.cold>:
  40148a:	0f 0b                	ud2

000000000040148c <uw_install_context_1.cold>:
  40148c:	e8 01 fd ff ff       	call   401192 <abort>

0000000000401491 <read_encoded_value.cold>:
  401491:	e8 fc fc ff ff       	call   401192 <abort>

0000000000401496 <execute_stack_op.cold>:
  401496:	e8 f7 fc ff ff       	call   401192 <abort>
  40149b:	e8 f2 fc ff ff       	call   401192 <abort>

00000000004014a0 <uw_update_context_1.cold>:
  4014a0:	e8 ed fc ff ff       	call   401192 <abort>
  4014a5:	e8 e8 fc ff ff       	call   401192 <abort>

00000000004014aa <execute_cfa_program_specialized.cold>:
  4014aa:	e8 e3 fc ff ff       	call   401192 <abort>

00000000004014af <execute_cfa_program_generic.cold>:
  4014af:	e8 de fc ff ff       	call   401192 <abort>

00000000004014b4 <uw_frame_state_for.cold>:
  4014b4:	e8 d9 fc ff ff       	call   401192 <abort>

00000000004014b9 <uw_init_context_1.cold>:
  4014b9:	e8 d4 fc ff ff       	call   401192 <abort>

00000000004014be <_Unwind_RaiseException_Phase2.cold>:
  4014be:	e8 cf fc ff ff       	call   401192 <abort>

00000000004014c3 <_Unwind_ForcedUnwind_Phase2.cold>:
  4014c3:	e8 ca fc ff ff       	call   401192 <abort>

00000000004014c8 <_Unwind_GetGR.cold>:
  4014c8:	55                   	push   %rbp
  4014c9:	48 89 e5             	mov    %rsp,%rbp
  4014cc:	e8 c1 fc ff ff       	call   401192 <abort>

00000000004014d1 <_Unwind_SetGR.cold>:
  4014d1:	55                   	push   %rbp
  4014d2:	48 89 e5             	mov    %rsp,%rbp
  4014d5:	e8 b8 fc ff ff       	call   401192 <abort>

00000000004014da <_Unwind_RaiseException.cold>:
  4014da:	e8 b3 fc ff ff       	call   401192 <abort>

00000000004014df <_Unwind_Resume.cold>:
  4014df:	e8 ae fc ff ff       	call   401192 <abort>

00000000004014e4 <_Unwind_Resume_or_Rethrow.cold>:
  4014e4:	e8 a9 fc ff ff       	call   401192 <abort>

00000000004014e9 <_Unwind_Backtrace.cold>:
  4014e9:	e8 a4 fc ff ff       	call   401192 <abort>

00000000004014ee <read_encoded_value_with_base.cold>:
  4014ee:	55                   	push   %rbp
  4014ef:	48 89 e5             	mov    %rsp,%rbp
  4014f2:	e8 9b fc ff ff       	call   401192 <abort>

00000000004014f7 <fde_mixed_encoding_extract.cold>:
  4014f7:	e8 96 fc ff ff       	call   401192 <abort>

00000000004014fc <classify_object_over_fdes.cold>:
  4014fc:	e8 91 fc ff ff       	call   401192 <abort>

0000000000401501 <__deregister_frame_info_bases.part.0.cold>:
  401501:	e8 8c fc ff ff       	call   401192 <abort>

0000000000401506 <fde_single_encoding_extract.cold>:
  401506:	e8 87 fc ff ff       	call   401192 <abort>

000000000040150b <fde_single_encoding_compare.cold>:
  40150b:	e8 82 fc ff ff       	call   401192 <abort>

0000000000401510 <fde_mixed_encoding_compare.cold>:
  401510:	e8 7d fc ff ff       	call   401192 <abort>

0000000000401515 <add_fdes.isra.0.cold>:
  401515:	e8 78 fc ff ff       	call   401192 <abort>

000000000040151a <linear_search_fdes.cold>:
  40151a:	e8 73 fc ff ff       	call   401192 <abort>

000000000040151f <_Unwind_Find_FDE.cold>:
  40151f:	e8 6e fc ff ff       	call   401192 <abort>

0000000000401524 <base_of_encoded_value.cold>:
  401524:	55                   	push   %rbp
  401525:	48 89 e5             	mov    %rsp,%rbp
  401528:	e8 65 fc ff ff       	call   401192 <abort>

000000000040152d <read_encoded_value_with_base.cold>:
  40152d:	55                   	push   %rbp
  40152e:	48 89 e5             	mov    %rsp,%rbp
  401531:	e8 5c fc ff ff       	call   401192 <abort>

0000000000401536 <__gcc_personality_v0.cold>:
  401536:	e8 57 fc ff ff       	call   401192 <abort>
  40153b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401540 <btree_release_tree_recursively>:
  401540:	55                   	push   %rbp
  401541:	48 89 e5             	mov    %rsp,%rbp
  401544:	41 55                	push   %r13
  401546:	49 89 fd             	mov    %rdi,%r13
  401549:	48 89 f7             	mov    %rsi,%rdi
  40154c:	41 54                	push   %r12
  40154e:	53                   	push   %rbx
  40154f:	48 89 f3             	mov    %rsi,%rbx
  401552:	48 83 ec 08          	sub    $0x8,%rsp
  401556:	e8 c5 e9 07 00       	call   47ff20 <version_lock_lock_exclusive>
  40155b:	44 8b 63 0c          	mov    0xc(%rbx),%r12d
  40155f:	45 85 e4             	test   %r12d,%r12d
  401562:	75 25                	jne    401589 <btree_release_tree_recursively+0x49>
  401564:	8b 43 08             	mov    0x8(%rbx),%eax
  401567:	85 c0                	test   %eax,%eax
  401569:	74 1e                	je     401589 <btree_release_tree_recursively+0x49>
  40156b:	44 89 e0             	mov    %r12d,%eax
  40156e:	4c 89 ef             	mov    %r13,%rdi
  401571:	41 83 c4 01          	add    $0x1,%r12d
  401575:	48 c1 e0 04          	shl    $0x4,%rax
  401579:	48 8b 74 18 18       	mov    0x18(%rax,%rbx,1),%rsi
  40157e:	e8 bd ff ff ff       	call   401540 <btree_release_tree_recursively>
  401583:	44 3b 63 08          	cmp    0x8(%rbx),%r12d
  401587:	72 e2                	jb     40156b <btree_release_tree_recursively+0x2b>
  401589:	c7 43 0c 02 00 00 00 	movl   $0x2,0xc(%rbx)
  401590:	49 8d 55 08          	lea    0x8(%r13),%rdx
  401594:	49 8b 45 08          	mov    0x8(%r13),%rax
  401598:	48 89 43 18          	mov    %rax,0x18(%rbx)
  40159c:	f0 48 0f b1 1a       	lock cmpxchg %rbx,(%rdx)
  4015a1:	75 f5                	jne    401598 <btree_release_tree_recursively+0x58>
  4015a3:	48 83 c4 08          	add    $0x8,%rsp
  4015a7:	48 89 df             	mov    %rbx,%rdi
  4015aa:	5b                   	pop    %rbx
  4015ab:	41 5c                	pop    %r12
  4015ad:	41 5d                	pop    %r13
  4015af:	5d                   	pop    %rbp
  4015b0:	e9 eb ee 07 00       	jmp    4804a0 <version_lock_unlock_exclusive>
  4015b5:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  4015bc:	00 00 00 00 

00000000004015c0 <btree_destroy>:
  4015c0:	55                   	push   %rbp
  4015c1:	31 f6                	xor    %esi,%esi
  4015c3:	48 89 e5             	mov    %rsp,%rbp
  4015c6:	41 54                	push   %r12
  4015c8:	49 89 fc             	mov    %rdi,%r12
  4015cb:	53                   	push   %rbx
  4015cc:	48 87 37             	xchg   %rsi,(%rdi)
  4015cf:	48 85 f6             	test   %rsi,%rsi
  4015d2:	75 27                	jne    4015fb <btree_destroy+0x3b>
  4015d4:	49 8b 5c 24 08       	mov    0x8(%r12),%rbx
  4015d9:	48 85 db             	test   %rbx,%rbx
  4015dc:	74 18                	je     4015f6 <btree_destroy+0x36>
  4015de:	66 90                	xchg   %ax,%ax
  4015e0:	48 89 df             	mov    %rbx,%rdi
  4015e3:	48 8b 5b 18          	mov    0x18(%rbx),%rbx
  4015e7:	e8 44 96 00 00       	call   40ac30 <__free>
  4015ec:	49 89 5c 24 08       	mov    %rbx,0x8(%r12)
  4015f1:	48 85 db             	test   %rbx,%rbx
  4015f4:	75 ea                	jne    4015e0 <btree_destroy+0x20>
  4015f6:	5b                   	pop    %rbx
  4015f7:	41 5c                	pop    %r12
  4015f9:	5d                   	pop    %rbp
  4015fa:	c3                   	ret
  4015fb:	e8 40 ff ff ff       	call   401540 <btree_release_tree_recursively>
  401600:	eb d2                	jmp    4015d4 <btree_destroy+0x14>
  401602:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  401609:	00 00 00 00 
  40160d:	0f 1f 00             	nopl   (%rax)

0000000000401610 <release_registered_frames>:
  401610:	f3 0f 1e fa          	endbr64
  401614:	55                   	push   %rbp
  401615:	48 8d 3d 54 6c 0b 00 	lea    0xb6c54(%rip),%rdi        # 4b8270 <registered_frames>
  40161c:	48 89 e5             	mov    %rsp,%rbp
  40161f:	e8 9c ff ff ff       	call   4015c0 <btree_destroy>
  401624:	48 8d 3d 25 6c 0b 00 	lea    0xb6c25(%rip),%rdi        # 4b8250 <registered_objects>
  40162b:	e8 90 ff ff ff       	call   4015c0 <btree_destroy>
  401630:	c6 05 11 6c 0b 00 01 	movb   $0x1,0xb6c11(%rip)        # 4b8248 <in_shutdown>
  401637:	5d                   	pop    %rbp
  401638:	c3                   	ret
  401639:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401640 <main>:
  401640:	f3 0f 1e fa          	endbr64
  401644:	31 c0                	xor    %eax,%eax
  401646:	c3                   	ret
  401647:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  40164e:	00 00 00 
  401651:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  401658:	00 00 00 
  40165b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401660 <_IO_stdfiles_init>:
  401660:	f3 0f 1e fa          	endbr64
  401664:	48 8b 05 b5 0a 0b 00 	mov    0xb0ab5(%rip),%rax        # 4b2120 <_IO_list_all>
  40166b:	48 85 c0             	test   %rax,%rax
  40166e:	74 24                	je     401694 <_IO_stdfiles_init+0x34>
  401670:	48 8d 15 a9 0a 0b 00 	lea    0xb0aa9(%rip),%rdx        # 4b2120 <_IO_list_all>
  401677:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40167e:	00 00 
  401680:	48 89 90 b8 00 00 00 	mov    %rdx,0xb8(%rax)
  401687:	48 8d 50 68          	lea    0x68(%rax),%rdx
  40168b:	48 8b 40 68          	mov    0x68(%rax),%rax
  40168f:	48 85 c0             	test   %rax,%rax
  401692:	75 ec                	jne    401680 <_IO_stdfiles_init+0x20>
  401694:	c3                   	ret
  401695:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  40169c:	00 00 00 
  40169f:	90                   	nop

00000000004016a0 <_start>:
  4016a0:	f3 0f 1e fa          	endbr64
  4016a4:	31 ed                	xor    %ebp,%ebp
  4016a6:	49 89 d1             	mov    %rdx,%r9
  4016a9:	5e                   	pop    %rsi
  4016aa:	48 89 e2             	mov    %rsp,%rdx
  4016ad:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4016b1:	50                   	push   %rax
  4016b2:	54                   	push   %rsp
  4016b3:	45 31 c0             	xor    %r8d,%r8d
  4016b6:	31 c9                	xor    %ecx,%ecx
  4016b8:	48 c7 c7 40 16 40 00 	mov    $0x401640,%rdi
  4016bf:	67 e8 7b 30 00 00    	addr32 call 404740 <__libc_start_main>
  4016c5:	f4                   	hlt
  4016c6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4016cd:	00 00 00 

00000000004016d0 <_dl_relocate_static_pie>:
  4016d0:	f3 0f 1e fa          	endbr64
  4016d4:	c3                   	ret
  4016d5:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4016dc:	00 00 00 
  4016df:	90                   	nop

00000000004016e0 <deregister_tm_clones>:
  4016e0:	b8 c8 2a 4b 00       	mov    $0x4b2ac8,%eax
  4016e5:	48 3d c8 2a 4b 00    	cmp    $0x4b2ac8,%rax
  4016eb:	74 13                	je     401700 <deregister_tm_clones+0x20>
  4016ed:	b8 00 00 00 00       	mov    $0x0,%eax
  4016f2:	48 85 c0             	test   %rax,%rax
  4016f5:	74 09                	je     401700 <deregister_tm_clones+0x20>
  4016f7:	bf c8 2a 4b 00       	mov    $0x4b2ac8,%edi
  4016fc:	ff e0                	jmp    *%rax
  4016fe:	66 90                	xchg   %ax,%ax
  401700:	c3                   	ret
  401701:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  401708:	00 00 00 00 
  40170c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401710 <register_tm_clones>:
  401710:	be c8 2a 4b 00       	mov    $0x4b2ac8,%esi
  401715:	48 81 ee c8 2a 4b 00 	sub    $0x4b2ac8,%rsi
  40171c:	48 89 f0             	mov    %rsi,%rax
  40171f:	48 c1 ee 3f          	shr    $0x3f,%rsi
  401723:	48 c1 f8 03          	sar    $0x3,%rax
  401727:	48 01 c6             	add    %rax,%rsi
  40172a:	48 d1 fe             	sar    $1,%rsi
  40172d:	74 11                	je     401740 <register_tm_clones+0x30>
  40172f:	b8 00 00 00 00       	mov    $0x0,%eax
  401734:	48 85 c0             	test   %rax,%rax
  401737:	74 07                	je     401740 <register_tm_clones+0x30>
  401739:	bf c8 2a 4b 00       	mov    $0x4b2ac8,%edi
  40173e:	ff e0                	jmp    *%rax
  401740:	c3                   	ret
  401741:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  401748:	00 00 00 00 
  40174c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401750 <__do_global_dtors_aux>:
  401750:	f3 0f 1e fa          	endbr64
  401754:	80 3d 85 13 0b 00 00 	cmpb   $0x0,0xb1385(%rip)        # 4b2ae0 <completed.1>
  40175b:	75 2b                	jne    401788 <__do_global_dtors_aux+0x38>
  40175d:	55                   	push   %rbp
  40175e:	48 89 e5             	mov    %rsp,%rbp
  401761:	e8 7a ff ff ff       	call   4016e0 <deregister_tm_clones>
  401766:	b8 30 27 48 00       	mov    $0x482730,%eax
  40176b:	48 85 c0             	test   %rax,%rax
  40176e:	74 0a                	je     40177a <__do_global_dtors_aux+0x2a>
  401770:	bf 80 24 4a 00       	mov    $0x4a2480,%edi
  401775:	e8 b6 0f 08 00       	call   482730 <__deregister_frame_info>
  40177a:	c6 05 5f 13 0b 00 01 	movb   $0x1,0xb135f(%rip)        # 4b2ae0 <completed.1>
  401781:	5d                   	pop    %rbp
  401782:	c3                   	ret
  401783:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401788:	c3                   	ret
  401789:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401790 <frame_dummy>:
  401790:	f3 0f 1e fa          	endbr64
  401794:	b8 80 24 48 00       	mov    $0x482480,%eax
  401799:	48 85 c0             	test   %rax,%rax
  40179c:	74 22                	je     4017c0 <frame_dummy+0x30>
  40179e:	55                   	push   %rbp
  40179f:	be 00 2b 4b 00       	mov    $0x4b2b00,%esi
  4017a4:	bf 80 24 4a 00       	mov    $0x4a2480,%edi
  4017a9:	48 89 e5             	mov    %rsp,%rbp
  4017ac:	e8 cf 0c 08 00       	call   482480 <__register_frame_info>
  4017b1:	5d                   	pop    %rbp
  4017b2:	e9 59 ff ff ff       	jmp    401710 <register_tm_clones>
  4017b7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4017be:	00 00 
  4017c0:	e9 4b ff ff ff       	jmp    401710 <register_tm_clones>
  4017c5:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4017cc:	00 00 00 
  4017cf:	90                   	nop

00000000004017d0 <f>:
  4017d0:	f3 0f 1e fa          	endbr64
  4017d4:	f2 0f 10 5f 08       	movsd  0x8(%rdi),%xmm3
  4017d9:	f2 0f 10 17          	movsd  (%rdi),%xmm2
  4017dd:	53                   	push   %rbx
  4017de:	66 0f ef c9          	pxor   %xmm1,%xmm1
  4017e2:	f2 0f 10 05 26 48 08 	movsd  0x84826(%rip),%xmm0        # 486010 <__rseq_flags+0xc>
  4017e9:	00 
  4017ea:	48 89 fb             	mov    %rdi,%rbx
  4017ed:	e8 6e 00 00 00       	call   401860 <__divdc3>
  4017f2:	f2 0f 10 5b 18       	movsd  0x18(%rbx),%xmm3
  4017f7:	f2 0f 10 53 10       	movsd  0x10(%rbx),%xmm2
  4017fc:	e8 5f 00 00 00       	call   401860 <__divdc3>
  401801:	f2 0f 10 5b 28       	movsd  0x28(%rbx),%xmm3
  401806:	f2 0f 10 53 20       	movsd  0x20(%rbx),%xmm2
  40180b:	e8 50 00 00 00       	call   401860 <__divdc3>
  401810:	f2 0f 10 5b 38       	movsd  0x38(%rbx),%xmm3
  401815:	f2 0f 10 53 30       	movsd  0x30(%rbx),%xmm2
  40181a:	e8 41 00 00 00       	call   401860 <__divdc3>
  40181f:	f2 0f 10 5b 48       	movsd  0x48(%rbx),%xmm3
  401824:	f2 0f 10 53 40       	movsd  0x40(%rbx),%xmm2
  401829:	e8 32 00 00 00       	call   401860 <__divdc3>
  40182e:	f2 0f 10 5b 58       	movsd  0x58(%rbx),%xmm3
  401833:	f2 0f 10 53 50       	movsd  0x50(%rbx),%xmm2
  401838:	e8 23 00 00 00       	call   401860 <__divdc3>
  40183d:	f2 0f 10 5b 68       	movsd  0x68(%rbx),%xmm3
  401842:	f2 0f 10 53 60       	movsd  0x60(%rbx),%xmm2
  401847:	e8 14 00 00 00       	call   401860 <__divdc3>
  40184c:	f2 0f 10 5b 78       	movsd  0x78(%rbx),%xmm3
  401851:	f2 0f 10 53 70       	movsd  0x70(%rbx),%xmm2
  401856:	5b                   	pop    %rbx
  401857:	e9 04 00 00 00       	jmp    401860 <__divdc3>
  40185c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401860 <__divdc3>:
  401860:	f3 0f 1e fa          	endbr64
  401864:	66 0f 28 f0          	movapd %xmm0,%xmm6
  401868:	f3 0f 7e 05 20 48 08 	movq   0x84820(%rip),%xmm0        # 486090 <conversion_rate+0x18>
  40186f:	00 
  401870:	66 0f 28 e2          	movapd %xmm2,%xmm4
  401874:	66 0f 28 eb          	movapd %xmm3,%xmm5
  401878:	66 0f 28 f9          	movapd %xmm1,%xmm7
  40187c:	66 0f 54 e0          	andpd  %xmm0,%xmm4
  401880:	66 0f 54 e8          	andpd  %xmm0,%xmm5
  401884:	66 0f 2f ec          	comisd %xmm4,%xmm5
  401888:	0f 86 c2 00 00 00    	jbe    401950 <__divdc3+0xf0>
  40188e:	66 0f 2f 2d 82 47 08 	comisd 0x84782(%rip),%xmm5        # 486018 <__rseq_flags+0x14>
  401895:	00 
  401896:	72 20                	jb     4018b8 <__divdc3+0x58>
  401898:	f2 0f 10 0d 80 47 08 	movsd  0x84780(%rip),%xmm1        # 486020 <__rseq_flags+0x1c>
  40189f:	00 
  4018a0:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  4018a4:	f2 0f 59 f1          	mulsd  %xmm1,%xmm6
  4018a8:	f2 0f 59 f9          	mulsd  %xmm1,%xmm7
  4018ac:	f2 0f 59 d1          	mulsd  %xmm1,%xmm2
  4018b0:	66 0f 28 eb          	movapd %xmm3,%xmm5
  4018b4:	66 0f 54 e8          	andpd  %xmm0,%xmm5
  4018b8:	f2 0f 10 0d 68 47 08 	movsd  0x84768(%rip),%xmm1        # 486028 <__rseq_flags+0x24>
  4018bf:	00 
  4018c0:	66 0f 2f cd          	comisd %xmm5,%xmm1
  4018c4:	0f 86 16 02 00 00    	jbe    401ae0 <__divdc3+0x280>
  4018ca:	f2 0f 10 0d 5e 47 08 	movsd  0x8475e(%rip),%xmm1        # 486030 <__rseq_flags+0x2c>
  4018d1:	00 
  4018d2:	f2 44 0f 10 05 5d 47 	movsd  0x8475d(%rip),%xmm8        # 486038 <__rseq_flags+0x34>
  4018d9:	08 00 
  4018db:	f2 0f 59 f1          	mulsd  %xmm1,%xmm6
  4018df:	f2 0f 59 f9          	mulsd  %xmm1,%xmm7
  4018e3:	f2 0f 59 d1          	mulsd  %xmm1,%xmm2
  4018e7:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  4018eb:	66 0f 28 e2          	movapd %xmm2,%xmm4
  4018ef:	66 44 0f 28 ca       	movapd %xmm2,%xmm9
  4018f4:	f2 0f 5e e3          	divsd  %xmm3,%xmm4
  4018f8:	f2 44 0f 59 cc       	mulsd  %xmm4,%xmm9
  4018fd:	66 0f 28 cc          	movapd %xmm4,%xmm1
  401901:	66 0f 54 c8          	andpd  %xmm0,%xmm1
  401905:	66 41 0f 2f c8       	comisd %xmm8,%xmm1
  40190a:	66 0f 28 ce          	movapd %xmm6,%xmm1
  40190e:	f2 44 0f 58 cb       	addsd  %xmm3,%xmm9
  401913:	0f 86 67 02 00 00    	jbe    401b80 <__divdc3+0x320>
  401919:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  40191d:	f2 0f 59 e7          	mulsd  %xmm7,%xmm4
  401921:	f2 0f 58 cf          	addsd  %xmm7,%xmm1
  401925:	f2 0f 5c e6          	subsd  %xmm6,%xmm4
  401929:	66 0f 28 e9          	movapd %xmm1,%xmm5
  40192d:	f2 41 0f 5e e9       	divsd  %xmm9,%xmm5
  401932:	f2 41 0f 5e e1       	divsd  %xmm9,%xmm4
  401937:	66 0f 2e ed          	ucomisd %xmm5,%xmm5
  40193b:	66 0f 28 cc          	movapd %xmm4,%xmm1
  40193f:	0f 8a c2 00 00 00    	jp     401a07 <__divdc3+0x1a7>
  401945:	66 0f 28 c5          	movapd %xmm5,%xmm0
  401949:	c3                   	ret
  40194a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401950:	66 0f 2f 25 c0 46 08 	comisd 0x846c0(%rip),%xmm4        # 486018 <__rseq_flags+0x14>
  401957:	00 
  401958:	72 20                	jb     40197a <__divdc3+0x11a>
  40195a:	f2 0f 10 0d be 46 08 	movsd  0x846be(%rip),%xmm1        # 486020 <__rseq_flags+0x1c>
  401961:	00 
  401962:	f2 0f 59 d1          	mulsd  %xmm1,%xmm2
  401966:	f2 0f 59 f1          	mulsd  %xmm1,%xmm6
  40196a:	f2 0f 59 f9          	mulsd  %xmm1,%xmm7
  40196e:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  401972:	66 0f 28 e2          	movapd %xmm2,%xmm4
  401976:	66 0f 54 e0          	andpd  %xmm0,%xmm4
  40197a:	f2 0f 10 0d a6 46 08 	movsd  0x846a6(%rip),%xmm1        # 486028 <__rseq_flags+0x24>
  401981:	00 
  401982:	66 0f 2f cc          	comisd %xmm4,%xmm1
  401986:	0f 86 ec 00 00 00    	jbe    401a78 <__divdc3+0x218>
  40198c:	f2 0f 10 0d 9c 46 08 	movsd  0x8469c(%rip),%xmm1        # 486030 <__rseq_flags+0x2c>
  401993:	00 
  401994:	f2 44 0f 10 05 9b 46 	movsd  0x8469b(%rip),%xmm8        # 486038 <__rseq_flags+0x34>
  40199b:	08 00 
  40199d:	f2 0f 59 f1          	mulsd  %xmm1,%xmm6
  4019a1:	f2 0f 59 f9          	mulsd  %xmm1,%xmm7
  4019a5:	f2 0f 59 d1          	mulsd  %xmm1,%xmm2
  4019a9:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  4019ad:	66 0f 28 cb          	movapd %xmm3,%xmm1
  4019b1:	66 44 0f 28 cb       	movapd %xmm3,%xmm9
  4019b6:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  4019ba:	f2 44 0f 59 c9       	mulsd  %xmm1,%xmm9
  4019bf:	66 0f 28 e1          	movapd %xmm1,%xmm4
  4019c3:	66 0f 54 e0          	andpd  %xmm0,%xmm4
  4019c7:	66 41 0f 2f e0       	comisd %xmm8,%xmm4
  4019cc:	f2 44 0f 58 ca       	addsd  %xmm2,%xmm9
  4019d1:	0f 86 69 01 00 00    	jbe    401b40 <__divdc3+0x2e0>
  4019d7:	66 0f 28 e1          	movapd %xmm1,%xmm4
  4019db:	66 0f 28 ef          	movapd %xmm7,%xmm5
  4019df:	f2 0f 59 e9          	mulsd  %xmm1,%xmm5
  4019e3:	66 0f 28 cf          	movapd %xmm7,%xmm1
  4019e7:	f2 0f 59 e6          	mulsd  %xmm6,%xmm4
  4019eb:	f2 0f 58 ee          	addsd  %xmm6,%xmm5
  4019ef:	f2 0f 5c cc          	subsd  %xmm4,%xmm1
  4019f3:	f2 41 0f 5e e9       	divsd  %xmm9,%xmm5
  4019f8:	f2 41 0f 5e c9       	divsd  %xmm9,%xmm1
  4019fd:	66 0f 2e ed          	ucomisd %xmm5,%xmm5
  401a01:	0f 8b 3e ff ff ff    	jnp    401945 <__divdc3+0xe5>
  401a07:	66 0f 2e c9          	ucomisd %xmm1,%xmm1
  401a0b:	0f 8b 34 ff ff ff    	jnp    401945 <__divdc3+0xe5>
  401a11:	66 0f ef e4          	pxor   %xmm4,%xmm4
  401a15:	ba 00 00 00 00       	mov    $0x0,%edx
  401a1a:	66 0f 2e d4          	ucomisd %xmm4,%xmm2
  401a1e:	0f 9b c0             	setnp  %al
  401a21:	0f 45 c2             	cmovne %edx,%eax
  401a24:	84 c0                	test   %al,%al
  401a26:	0f 84 84 01 00 00    	je     401bb0 <__divdc3+0x350>
  401a2c:	66 0f 2e dc          	ucomisd %xmm4,%xmm3
  401a30:	0f 9b c0             	setnp  %al
  401a33:	0f 45 c2             	cmovne %edx,%eax
  401a36:	84 c0                	test   %al,%al
  401a38:	0f 84 72 01 00 00    	je     401bb0 <__divdc3+0x350>
  401a3e:	66 0f 2e f6          	ucomisd %xmm6,%xmm6
  401a42:	7b 0a                	jnp    401a4e <__divdc3+0x1ee>
  401a44:	66 0f 2e ff          	ucomisd %xmm7,%xmm7
  401a48:	0f 8a f7 fe ff ff    	jp     401945 <__divdc3+0xe5>
  401a4e:	f3 0f 7e 0d 4a 46 08 	movq   0x8464a(%rip),%xmm1        # 4860a0 <conversion_rate+0x28>
  401a55:	00 
  401a56:	66 0f 54 ca          	andpd  %xmm2,%xmm1
  401a5a:	66 0f 56 0d 4e 46 08 	orpd   0x8464e(%rip),%xmm1        # 4860b0 <conversion_rate+0x38>
  401a61:	00 
  401a62:	f2 0f 59 f1          	mulsd  %xmm1,%xmm6
  401a66:	f2 0f 59 cf          	mulsd  %xmm7,%xmm1
  401a6a:	66 0f 28 ee          	movapd %xmm6,%xmm5
  401a6e:	66 0f 28 c5          	movapd %xmm5,%xmm0
  401a72:	c3                   	ret
  401a73:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401a78:	f2 44 0f 10 05 b7 45 	movsd  0x845b7(%rip),%xmm8        # 486038 <__rseq_flags+0x34>
  401a7f:	08 00 
  401a81:	66 0f 28 ee          	movapd %xmm6,%xmm5
  401a85:	66 44 0f 28 cf       	movapd %xmm7,%xmm9
  401a8a:	66 0f 54 e8          	andpd  %xmm0,%xmm5
  401a8e:	66 44 0f 54 c8       	andpd  %xmm0,%xmm9
  401a93:	66 44 0f 2f c5       	comisd %xmm5,%xmm8
  401a98:	0f 86 d2 02 00 00    	jbe    401d70 <__divdc3+0x510>
  401a9e:	f2 0f 10 0d 9a 45 08 	movsd  0x8459a(%rip),%xmm1        # 486040 <__rseq_flags+0x3c>
  401aa5:	00 
  401aa6:	66 41 0f 2f c9       	comisd %xmm9,%xmm1
  401aab:	0f 86 fc fe ff ff    	jbe    4019ad <__divdc3+0x14d>
  401ab1:	66 0f 2f cc          	comisd %xmm4,%xmm1
  401ab5:	0f 86 f2 fe ff ff    	jbe    4019ad <__divdc3+0x14d>
  401abb:	f2 0f 10 0d 6d 45 08 	movsd  0x8456d(%rip),%xmm1        # 486030 <__rseq_flags+0x2c>
  401ac2:	00 
  401ac3:	f2 0f 59 f1          	mulsd  %xmm1,%xmm6
  401ac7:	f2 0f 59 f9          	mulsd  %xmm1,%xmm7
  401acb:	f2 0f 59 d1          	mulsd  %xmm1,%xmm2
  401acf:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  401ad3:	e9 d5 fe ff ff       	jmp    4019ad <__divdc3+0x14d>
  401ad8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401adf:	00 
  401ae0:	f2 44 0f 10 05 4f 45 	movsd  0x8454f(%rip),%xmm8        # 486038 <__rseq_flags+0x34>
  401ae7:	08 00 
  401ae9:	66 0f 28 e6          	movapd %xmm6,%xmm4
  401aed:	66 44 0f 28 cf       	movapd %xmm7,%xmm9
  401af2:	66 0f 54 e0          	andpd  %xmm0,%xmm4
  401af6:	66 44 0f 54 c8       	andpd  %xmm0,%xmm9
  401afb:	66 44 0f 2f c4       	comisd %xmm4,%xmm8
  401b00:	0f 86 92 02 00 00    	jbe    401d98 <__divdc3+0x538>
  401b06:	f2 0f 10 0d 32 45 08 	movsd  0x84532(%rip),%xmm1        # 486040 <__rseq_flags+0x3c>
  401b0d:	00 
  401b0e:	66 41 0f 2f c9       	comisd %xmm9,%xmm1
  401b13:	0f 86 d2 fd ff ff    	jbe    4018eb <__divdc3+0x8b>
  401b19:	66 0f 2f cd          	comisd %xmm5,%xmm1
  401b1d:	0f 86 c8 fd ff ff    	jbe    4018eb <__divdc3+0x8b>
  401b23:	f2 0f 10 0d 05 45 08 	movsd  0x84505(%rip),%xmm1        # 486030 <__rseq_flags+0x2c>
  401b2a:	00 
  401b2b:	f2 0f 59 f1          	mulsd  %xmm1,%xmm6
  401b2f:	f2 0f 59 f9          	mulsd  %xmm1,%xmm7
  401b33:	f2 0f 59 d1          	mulsd  %xmm1,%xmm2
  401b37:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  401b3b:	e9 ab fd ff ff       	jmp    4018eb <__divdc3+0x8b>
  401b40:	66 0f 28 cf          	movapd %xmm7,%xmm1
  401b44:	66 0f 28 e6          	movapd %xmm6,%xmm4
  401b48:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  401b4c:	f2 0f 5e e2          	divsd  %xmm2,%xmm4
  401b50:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  401b54:	f2 0f 58 ce          	addsd  %xmm6,%xmm1
  401b58:	66 0f 28 e9          	movapd %xmm1,%xmm5
  401b5c:	66 0f 28 cf          	movapd %xmm7,%xmm1
  401b60:	f2 41 0f 5e e9       	divsd  %xmm9,%xmm5
  401b65:	f2 0f 59 e3          	mulsd  %xmm3,%xmm4
  401b69:	f2 0f 5c cc          	subsd  %xmm4,%xmm1
  401b6d:	f2 41 0f 5e c9       	divsd  %xmm9,%xmm1
  401b72:	e9 86 fe ff ff       	jmp    4019fd <__divdc3+0x19d>
  401b77:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  401b7e:	00 00 
  401b80:	f2 0f 5e cb          	divsd  %xmm3,%xmm1
  401b84:	f2 0f 59 ca          	mulsd  %xmm2,%xmm1
  401b88:	f2 0f 58 cf          	addsd  %xmm7,%xmm1
  401b8c:	66 0f 28 e9          	movapd %xmm1,%xmm5
  401b90:	66 0f 28 cf          	movapd %xmm7,%xmm1
  401b94:	f2 0f 5e cb          	divsd  %xmm3,%xmm1
  401b98:	f2 41 0f 5e e9       	divsd  %xmm9,%xmm5
  401b9d:	f2 0f 59 ca          	mulsd  %xmm2,%xmm1
  401ba1:	f2 0f 5c ce          	subsd  %xmm6,%xmm1
  401ba5:	f2 41 0f 5e c9       	divsd  %xmm9,%xmm1
  401baa:	e9 4e fe ff ff       	jmp    4019fd <__divdc3+0x19d>
  401baf:	90                   	nop
  401bb0:	f2 0f 10 25 90 44 08 	movsd  0x84490(%rip),%xmm4        # 486048 <__rseq_flags+0x44>
  401bb7:	00 
  401bb8:	66 44 0f 28 ce       	movapd %xmm6,%xmm9
  401bbd:	66 44 0f 28 c2       	movapd %xmm2,%xmm8
  401bc2:	66 44 0f 54 c8       	andpd  %xmm0,%xmm9
  401bc7:	66 44 0f 54 c0       	andpd  %xmm0,%xmm8
  401bcc:	66 44 0f 2e cc       	ucomisd %xmm4,%xmm9
  401bd1:	0f 87 d9 00 00 00    	ja     401cb0 <__divdc3+0x450>
  401bd7:	66 44 0f 28 d7       	movapd %xmm7,%xmm10
  401bdc:	66 44 0f 54 d0       	andpd  %xmm0,%xmm10
  401be1:	66 44 0f 2e d4       	ucomisd %xmm4,%xmm10
  401be6:	0f 87 c4 00 00 00    	ja     401cb0 <__divdc3+0x450>
  401bec:	66 44 0f 2e c4       	ucomisd %xmm4,%xmm8
  401bf1:	0f 86 c9 01 00 00    	jbe    401dc0 <__divdc3+0x560>
  401bf7:	f2 44 0f 10 05 b0 44 	movsd  0x844b0(%rip),%xmm8        # 4860b0 <conversion_rate+0x38>
  401bfe:	08 00 
  401c00:	66 41 0f 2e e1       	ucomisd %xmm9,%xmm4
  401c05:	0f 82 3a fd ff ff    	jb     401945 <__divdc3+0xe5>
  401c0b:	66 41 0f 2e e2       	ucomisd %xmm10,%xmm4
  401c10:	0f 82 2f fd ff ff    	jb     401945 <__divdc3+0xe5>
  401c16:	31 c0                	xor    %eax,%eax
  401c18:	66 0f ef ed          	pxor   %xmm5,%xmm5
  401c1c:	f3 0f 7e 0d 7c 44 08 	movq   0x8447c(%rip),%xmm1        # 4860a0 <conversion_rate+0x28>
  401c23:	00 
  401c24:	66 0f 54 c3          	andpd  %xmm3,%xmm0
  401c28:	66 44 0f 2e 05 17 44 	ucomisd 0x84417(%rip),%xmm8        # 486048 <__rseq_flags+0x44>
  401c2f:	08 00 
  401c31:	66 0f 28 e1          	movapd %xmm1,%xmm4
  401c35:	66 0f 54 d1          	andpd  %xmm1,%xmm2
  401c39:	0f 97 c0             	seta   %al
  401c3c:	f2 0f 2a e8          	cvtsi2sd %eax,%xmm5
  401c40:	31 c0                	xor    %eax,%eax
  401c42:	66 0f 2e 05 fe 43 08 	ucomisd 0x843fe(%rip),%xmm0        # 486048 <__rseq_flags+0x44>
  401c49:	00 
  401c4a:	66 0f 28 c1          	movapd %xmm1,%xmm0
  401c4e:	66 0f 54 cb          	andpd  %xmm3,%xmm1
  401c52:	66 0f 28 df          	movapd %xmm7,%xmm3
  401c56:	66 0f 55 e5          	andnpd %xmm5,%xmm4
  401c5a:	0f 97 c0             	seta   %al
  401c5d:	66 0f 56 d4          	orpd   %xmm4,%xmm2
  401c61:	66 0f ef e4          	pxor   %xmm4,%xmm4
  401c65:	f2 0f 2a e0          	cvtsi2sd %eax,%xmm4
  401c69:	f2 0f 59 fa          	mulsd  %xmm2,%xmm7
  401c6d:	66 0f 55 c4          	andnpd %xmm4,%xmm0
  401c71:	66 0f ef e4          	pxor   %xmm4,%xmm4
  401c75:	66 0f 56 c1          	orpd   %xmm1,%xmm0
  401c79:	66 0f 28 ce          	movapd %xmm6,%xmm1
  401c7d:	f2 0f 59 ca          	mulsd  %xmm2,%xmm1
  401c81:	f2 0f 59 d8          	mulsd  %xmm0,%xmm3
  401c85:	f2 0f 59 f0          	mulsd  %xmm0,%xmm6
  401c89:	f2 0f 58 cb          	addsd  %xmm3,%xmm1
  401c8d:	66 0f 28 e9          	movapd %xmm1,%xmm5
  401c91:	66 0f 28 cf          	movapd %xmm7,%xmm1
  401c95:	f2 0f 5c ce          	subsd  %xmm6,%xmm1
  401c99:	f2 0f 59 ec          	mulsd  %xmm4,%xmm5
  401c9d:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  401ca1:	e9 9f fc ff ff       	jmp    401945 <__divdc3+0xe5>
  401ca6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  401cad:	00 00 00 
  401cb0:	66 41 0f 2e e0       	ucomisd %xmm8,%xmm4
  401cb5:	0f 82 8a fc ff ff    	jb     401945 <__divdc3+0xe5>
  401cbb:	66 44 0f 28 c3       	movapd %xmm3,%xmm8
  401cc0:	66 44 0f 54 c0       	andpd  %xmm0,%xmm8
  401cc5:	66 41 0f 2e e0       	ucomisd %xmm8,%xmm4
  401cca:	0f 82 75 fc ff ff    	jb     401945 <__divdc3+0xe5>
  401cd0:	31 c0                	xor    %eax,%eax
  401cd2:	66 0f ef e4          	pxor   %xmm4,%xmm4
  401cd6:	f3 0f 7e 0d c2 43 08 	movq   0x843c2(%rip),%xmm1        # 4860a0 <conversion_rate+0x28>
  401cdd:	00 
  401cde:	66 0f 54 c7          	andpd  %xmm7,%xmm0
  401ce2:	66 44 0f 2e 0d 5d 43 	ucomisd 0x8435d(%rip),%xmm9        # 486048 <__rseq_flags+0x44>
  401ce9:	08 00 
  401ceb:	66 0f 28 e9          	movapd %xmm1,%xmm5
  401cef:	0f 97 c0             	seta   %al
  401cf2:	f2 0f 2a e0          	cvtsi2sd %eax,%xmm4
  401cf6:	31 c0                	xor    %eax,%eax
  401cf8:	66 0f 2e 05 48 43 08 	ucomisd 0x84348(%rip),%xmm0        # 486048 <__rseq_flags+0x44>
  401cff:	00 
  401d00:	66 0f 28 c1          	movapd %xmm1,%xmm0
  401d04:	66 0f 55 ec          	andnpd %xmm4,%xmm5
  401d08:	66 0f 28 e6          	movapd %xmm6,%xmm4
  401d0c:	0f 97 c0             	seta   %al
  401d0f:	66 0f 54 e1          	andpd  %xmm1,%xmm4
  401d13:	66 0f 54 cf          	andpd  %xmm7,%xmm1
  401d17:	66 0f 56 ec          	orpd   %xmm4,%xmm5
  401d1b:	66 0f ef e4          	pxor   %xmm4,%xmm4
  401d1f:	f2 0f 2a e0          	cvtsi2sd %eax,%xmm4
  401d23:	66 0f 28 f5          	movapd %xmm5,%xmm6
  401d27:	66 0f 55 c4          	andnpd %xmm4,%xmm0
  401d2b:	66 0f 28 e3          	movapd %xmm3,%xmm4
  401d2f:	f2 0f 59 de          	mulsd  %xmm6,%xmm3
  401d33:	66 0f 56 c1          	orpd   %xmm1,%xmm0
  401d37:	66 0f 28 ca          	movapd %xmm2,%xmm1
  401d3b:	f2 0f 59 cd          	mulsd  %xmm5,%xmm1
  401d3f:	f2 0f 59 e0          	mulsd  %xmm0,%xmm4
  401d43:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  401d47:	f2 0f 58 cc          	addsd  %xmm4,%xmm1
  401d4b:	f2 0f 10 25 5d 43 08 	movsd  0x8435d(%rip),%xmm4        # 4860b0 <conversion_rate+0x38>
  401d52:	00 
  401d53:	66 0f 28 e9          	movapd %xmm1,%xmm5
  401d57:	66 0f 28 ca          	movapd %xmm2,%xmm1
  401d5b:	f2 0f 5c cb          	subsd  %xmm3,%xmm1
  401d5f:	f2 0f 59 ec          	mulsd  %xmm4,%xmm5
  401d63:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  401d67:	e9 d9 fb ff ff       	jmp    401945 <__divdc3+0xe5>
  401d6c:	0f 1f 40 00          	nopl   0x0(%rax)
  401d70:	66 45 0f 2f c1       	comisd %xmm9,%xmm8
  401d75:	0f 86 32 fc ff ff    	jbe    4019ad <__divdc3+0x14d>
  401d7b:	f2 0f 10 0d bd 42 08 	movsd  0x842bd(%rip),%xmm1        # 486040 <__rseq_flags+0x3c>
  401d82:	00 
  401d83:	66 0f 2f cd          	comisd %xmm5,%xmm1
  401d87:	0f 86 20 fc ff ff    	jbe    4019ad <__divdc3+0x14d>
  401d8d:	e9 1f fd ff ff       	jmp    401ab1 <__divdc3+0x251>
  401d92:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401d98:	66 45 0f 2f c1       	comisd %xmm9,%xmm8
  401d9d:	0f 86 48 fb ff ff    	jbe    4018eb <__divdc3+0x8b>
  401da3:	f2 0f 10 0d 95 42 08 	movsd  0x84295(%rip),%xmm1        # 486040 <__rseq_flags+0x3c>
  401daa:	00 
  401dab:	66 0f 2f cc          	comisd %xmm4,%xmm1
  401daf:	0f 86 36 fb ff ff    	jbe    4018eb <__divdc3+0x8b>
  401db5:	e9 5f fd ff ff       	jmp    401b19 <__divdc3+0x2b9>
  401dba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401dc0:	66 44 0f 28 db       	movapd %xmm3,%xmm11
  401dc5:	66 44 0f 54 d8       	andpd  %xmm0,%xmm11
  401dca:	66 44 0f 2e dc       	ucomisd %xmm4,%xmm11
  401dcf:	0f 87 2b fe ff ff    	ja     401c00 <__divdc3+0x3a0>
  401dd5:	e9 6b fb ff ff       	jmp    401945 <__divdc3+0xe5>
  401dda:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
