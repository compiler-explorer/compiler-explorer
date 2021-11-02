
test:     file format elf64-x86-64


Disassembly of section .init:

0000000000401000 <_init>:
_init():
  401000:	f3 0f 1e fa          	endbr64 
  401004:	48 83 ec 08          	sub    rsp,0x8
  401008:	48 8b 05 e9 2f 00 00 	mov    rax,QWORD PTR [rip+0x2fe9]        # 403ff8 <__gmon_start__>
  40100f:	48 85 c0             	test   rax,rax
  401012:	74 02                	je     401016 <_init+0x16>
  401014:	ff d0                	call   rax
  401016:	48 83 c4 08          	add    rsp,0x8
  40101a:	c3                   	ret    

Disassembly of section .text:

0000000000401020 <main>:
main():
/app/main.cpp:4
  401020:	bf 64 00 00 00       	mov    edi,0x64
  401025:	e9 f6 00 00 00       	jmp    401120 <_Z6squarei>
  40102a:	66 0f 1f 44 00 00    	nop    WORD PTR [rax+rax*1+0x0]

0000000000401030 <_start>:
_start():
  401030:	f3 0f 1e fa          	endbr64 
  401034:	31 ed                	xor    ebp,ebp
  401036:	49 89 d1             	mov    r9,rdx
  401039:	5e                   	pop    rsi
  40103a:	48 89 e2             	mov    rdx,rsp
  40103d:	48 83 e4 f0          	and    rsp,0xfffffffffffffff0
  401041:	50                   	push   rax
  401042:	54                   	push   rsp
  401043:	49 c7 c0 a0 11 40 00 	mov    r8,0x4011a0
  40104a:	48 c7 c1 30 11 40 00 	mov    rcx,0x401130
  401051:	48 c7 c7 20 10 40 00 	mov    rdi,0x401020
  401058:	ff 15 92 2f 00 00    	call   QWORD PTR [rip+0x2f92]        # 403ff0 <__libc_start_main@GLIBC_2.2.5>
  40105e:	f4                   	hlt    
  40105f:	90                   	nop

0000000000401060 <_dl_relocate_static_pie>:
_dl_relocate_static_pie():
  401060:	f3 0f 1e fa          	endbr64 
  401064:	c3                   	ret    
  401065:	66 2e 0f 1f 84 00 00 	cs nop WORD PTR [rax+rax*1+0x0]
  40106c:	00 00 00 
  40106f:	90                   	nop

0000000000401070 <deregister_tm_clones>:
deregister_tm_clones():
  401070:	b8 28 40 40 00       	mov    eax,0x404028
  401075:	48 3d 28 40 40 00    	cmp    rax,0x404028
  40107b:	74 13                	je     401090 <deregister_tm_clones+0x20>
  40107d:	b8 00 00 00 00       	mov    eax,0x0
  401082:	48 85 c0             	test   rax,rax
  401085:	74 09                	je     401090 <deregister_tm_clones+0x20>
  401087:	bf 28 40 40 00       	mov    edi,0x404028
  40108c:	ff e0                	jmp    rax
  40108e:	66 90                	xchg   ax,ax
  401090:	c3                   	ret    
  401091:	66 66 2e 0f 1f 84 00 	data16 cs nop WORD PTR [rax+rax*1+0x0]
  401098:	00 00 00 00 
  40109c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]

00000000004010a0 <register_tm_clones>:
register_tm_clones():
  4010a0:	be 28 40 40 00       	mov    esi,0x404028
  4010a5:	48 81 ee 28 40 40 00 	sub    rsi,0x404028
  4010ac:	48 89 f0             	mov    rax,rsi
  4010af:	48 c1 ee 3f          	shr    rsi,0x3f
  4010b3:	48 c1 f8 03          	sar    rax,0x3
  4010b7:	48 01 c6             	add    rsi,rax
  4010ba:	48 d1 fe             	sar    rsi,1
  4010bd:	74 11                	je     4010d0 <register_tm_clones+0x30>
  4010bf:	b8 00 00 00 00       	mov    eax,0x0
  4010c4:	48 85 c0             	test   rax,rax
  4010c7:	74 07                	je     4010d0 <register_tm_clones+0x30>
  4010c9:	bf 28 40 40 00       	mov    edi,0x404028
  4010ce:	ff e0                	jmp    rax
  4010d0:	c3                   	ret    
  4010d1:	66 66 2e 0f 1f 84 00 	data16 cs nop WORD PTR [rax+rax*1+0x0]
  4010d8:	00 00 00 00 
  4010dc:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]

00000000004010e0 <__do_global_dtors_aux>:
__do_global_dtors_aux():
  4010e0:	f3 0f 1e fa          	endbr64 
  4010e4:	80 3d 3d 2f 00 00 00 	cmp    BYTE PTR [rip+0x2f3d],0x0        # 404028 <completed.0>
  4010eb:	75 13                	jne    401100 <__do_global_dtors_aux+0x20>
  4010ed:	55                   	push   rbp
  4010ee:	48 89 e5             	mov    rbp,rsp
  4010f1:	e8 7a ff ff ff       	call   401070 <deregister_tm_clones>
  4010f6:	c6 05 2b 2f 00 00 01 	mov    BYTE PTR [rip+0x2f2b],0x1        # 404028 <completed.0>
  4010fd:	5d                   	pop    rbp
  4010fe:	c3                   	ret    
  4010ff:	90                   	nop
  401100:	c3                   	ret    
  401101:	66 66 2e 0f 1f 84 00 	data16 cs nop WORD PTR [rax+rax*1+0x0]
  401108:	00 00 00 00 
  40110c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]

0000000000401110 <frame_dummy>:
frame_dummy():
  401110:	f3 0f 1e fa          	endbr64 
  401114:	eb 8a                	jmp    4010a0 <register_tm_clones>
  401116:	66 2e 0f 1f 84 00 00 	cs nop WORD PTR [rax+rax*1+0x0]
  40111d:	00 00 00 

0000000000401120 <_Z6squarei>:
_Z6squarei():
/app/example.cpp:5
  401120:	89 f8                	mov    eax,edi
  401122:	0f af c7             	imul   eax,edi
/app/example.cpp:6
  401125:	c3                   	ret    
  401126:	66 2e 0f 1f 84 00 00 	cs nop WORD PTR [rax+rax*1+0x0]
  40112d:	00 00 00 

0000000000401130 <__libc_csu_init>:
__libc_csu_init():
  401130:	f3 0f 1e fa          	endbr64 
  401134:	41 57                	push   r15
  401136:	4c 8d 3d c3 2c 00 00 	lea    r15,[rip+0x2cc3]        # 403e00 <__frame_dummy_init_array_entry>
  40113d:	41 56                	push   r14
  40113f:	49 89 d6             	mov    r14,rdx
  401142:	41 55                	push   r13
  401144:	49 89 f5             	mov    r13,rsi
  401147:	41 54                	push   r12
  401149:	41 89 fc             	mov    r12d,edi
  40114c:	55                   	push   rbp
  40114d:	48 8d 2d b4 2c 00 00 	lea    rbp,[rip+0x2cb4]        # 403e08 <__do_global_dtors_aux_fini_array_entry>
  401154:	53                   	push   rbx
  401155:	4c 29 fd             	sub    rbp,r15
  401158:	48 83 ec 08          	sub    rsp,0x8
  40115c:	e8 9f fe ff ff       	call   401000 <_init>
  401161:	48 c1 fd 03          	sar    rbp,0x3
  401165:	74 1f                	je     401186 <__libc_csu_init+0x56>
  401167:	31 db                	xor    ebx,ebx
  401169:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
  401170:	4c 89 f2             	mov    rdx,r14
  401173:	4c 89 ee             	mov    rsi,r13
  401176:	44 89 e7             	mov    edi,r12d
  401179:	41 ff 14 df          	call   QWORD PTR [r15+rbx*8]
  40117d:	48 83 c3 01          	add    rbx,0x1
  401181:	48 39 dd             	cmp    rbp,rbx
  401184:	75 ea                	jne    401170 <__libc_csu_init+0x40>
  401186:	48 83 c4 08          	add    rsp,0x8
  40118a:	5b                   	pop    rbx
  40118b:	5d                   	pop    rbp
  40118c:	41 5c                	pop    r12
  40118e:	41 5d                	pop    r13
  401190:	41 5e                	pop    r14
  401192:	41 5f                	pop    r15
  401194:	c3                   	ret    
  401195:	66 66 2e 0f 1f 84 00 	data16 cs nop WORD PTR [rax+rax*1+0x0]
  40119c:	00 00 00 00 

00000000004011a0 <__libc_csu_fini>:
__libc_csu_fini():
  4011a0:	f3 0f 1e fa          	endbr64 
  4011a4:	c3                   	ret    

Disassembly of section .fini:

00000000004011a8 <_fini>:
_fini():
  4011a8:	f3 0f 1e fa          	endbr64 
  4011ac:	48 83 ec 08          	sub    rsp,0x8
  4011b0:	48 83 c4 08          	add    rsp,0x8
  4011b4:	c3                   	ret    
