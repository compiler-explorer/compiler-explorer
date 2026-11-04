
C:\Windows\TEMP\compiler-explorer-compilerCLAUDEclangcl\output.s.exe:     file format pei-x86-64


Disassembly of section .text:

0000000140001000 <.text>:
   140001000:	89 c8                                           	mov    eax,ecx
   140001002:	0f af c1                                        	imul   eax,ecx
   140001005:	c3                                              	ret
   140001006:	48 83 ec 28                                     	sub    rsp,0x28
   14000100a:	48 8d 0d ef 1f 00 00                            	lea    rcx,[rip+0x1fef]        # 0x140003000
   140001011:	ba 31 00 00 00                                  	mov    edx,0x31
   140001016:	e8 07 00 00 00                                  	call   0x140001022
   14000101b:	31 c0                                           	xor    eax,eax
   14000101d:	48 83 c4 28                                     	add    rsp,0x28
   140001021:	c3                                              	ret
   140001022:	56                                              	push   rsi
   140001023:	57                                              	push   rdi
   140001024:	53                                              	push   rbx
   140001025:	48 83 ec 40                                     	sub    rsp,0x40
   140001029:	48 89 ce                                        	mov    rsi,rcx
   14000102c:	48 8d 5c 24 68                                  	lea    rbx,[rsp+0x68]
   140001031:	48 89 13                                        	mov    QWORD PTR [rbx],rdx
   140001034:	4c 89 43 08                                     	mov    QWORD PTR [rbx+0x8],r8
   140001038:	4c 89 4b 10                                     	mov    QWORD PTR [rbx+0x10],r9
   14000103c:	48 8b 05 bd 2f 00 00                            	mov    rax,QWORD PTR [rip+0x2fbd]        # 0x140004000
   140001043:	48 31 e0                                        	xor    rax,rsp
   140001046:	48 89 44 24 38                                  	mov    QWORD PTR [rsp+0x38],rax
   14000104b:	48 89 5c 24 30                                  	mov    QWORD PTR [rsp+0x30],rbx
   140001050:	b9 01 00 00 00                                  	mov    ecx,0x1
   140001055:	ff 15 dd 25 00 00                               	call   QWORD PTR [rip+0x25dd]        # 0x140003638
   14000105b:	48 89 c7                                        	mov    rdi,rax
   14000105e:	e8 41 00 00 00                                  	call   0x1400010a4
   140001063:	48 8b 08                                        	mov    rcx,QWORD PTR [rax]
   140001066:	48 89 5c 24 20                                  	mov    QWORD PTR [rsp+0x20],rbx
   14000106b:	48 89 fa                                        	mov    rdx,rdi
   14000106e:	49 89 f0                                        	mov    r8,rsi
   140001071:	45 31 c9                                        	xor    r9d,r9d
   140001074:	ff 15 ce 25 00 00                               	call   QWORD PTR [rip+0x25ce]        # 0x140003648
   14000107a:	48 8b 4c 24 38                                  	mov    rcx,QWORD PTR [rsp+0x38]
   14000107f:	48 31 e1                                        	xor    rcx,rsp
   140001082:	48 8b 15 77 2f 00 00                            	mov    rdx,QWORD PTR [rip+0x2f77]        # 0x140004000
   140001089:	48 39 ca                                        	cmp    rdx,rcx
   14000108c:	75 08                                           	jne    0x140001096
   14000108e:	48 83 c4 40                                     	add    rsp,0x40
   140001092:	5b                                              	pop    rbx
   140001093:	5f                                              	pop    rdi
   140001094:	5e                                              	pop    rsi
   140001095:	c3                                              	ret
   140001096:	48 8b 4c 24 38                                  	mov    rcx,QWORD PTR [rsp+0x38]
   14000109b:	48 31 e1                                        	xor    rcx,rsp
   14000109e:	e8 1d 00 00 00                                  	call   0x1400010c0
   1400010a3:	cc                                              	int3
   1400010a4:	48 8d 05 cd 2f 00 00                            	lea    rax,[rip+0x2fcd]        # 0x140004078
   1400010ab:	c3                                              	ret
   1400010ac:	cc                                              	int3
   1400010ad:	cc                                              	int3
   1400010ae:	cc                                              	int3
   1400010af:	cc                                              	int3
   1400010b0:	cc                                              	int3
   1400010b1:	cc                                              	int3
   1400010b2:	cc                                              	int3
   1400010b3:	cc                                              	int3
   1400010b4:	cc                                              	int3
   1400010b5:	cc                                              	int3
   1400010b6:	66 66 0f 1f 84 00 00 00 00 00                   	data16 nop WORD PTR [rax+rax*1+0x0]
   1400010c0:	48 3b 0d 39 2f 00 00                            	cmp    rcx,QWORD PTR [rip+0x2f39]        # 0x140004000
   1400010c7:	75 10                                           	jne    0x1400010d9
   1400010c9:	48 c1 c1 10                                     	rol    rcx,0x10
   1400010cd:	66 f7 c1 ff ff                                  	test   cx,0xffff
   1400010d2:	75 01                                           	jne    0x1400010d5
   1400010d4:	c3                                              	ret
   1400010d5:	48 c1 c9 10                                     	ror    rcx,0x10
   1400010d9:	e9 1a 00 00 00                                  	jmp    0x1400010f8
   1400010de:	cc                                              	int3
   1400010df:	cc                                              	int3
   1400010e0:	8b c9                                           	mov    ecx,ecx
   1400010e2:	cd 29                                           	int    0x29
   1400010e4:	c3                                              	ret
   1400010e5:	cc                                              	int3
   1400010e6:	cc                                              	int3
   1400010e7:	cc                                              	int3
   1400010e8:	8b c9                                           	mov    ecx,ecx
   1400010ea:	cd 29                                           	int    0x29
   1400010ec:	c3                                              	ret
   1400010ed:	cc                                              	int3
   1400010ee:	cc                                              	int3
   1400010ef:	cc                                              	int3
   1400010f0:	b9 08 00 00 00                                  	mov    ecx,0x8
   1400010f5:	cd 29                                           	int    0x29
   1400010f7:	c3                                              	ret
   1400010f8:	b9 02 00 00 00                                  	mov    ecx,0x2
   1400010fd:	cd 29                                           	int    0x29
   1400010ff:	c3                                              	ret
   140001100:	40 53                                           	rex push rbx
   140001102:	48 83 ec 20                                     	sub    rsp,0x20
   140001106:	b9 01 00 00 00                                  	mov    ecx,0x1
   14000110b:	e8 40 12 00 00                                  	call   0x140002350
   140001110:	e8 4b 08 00 00                                  	call   0x140001960
   140001115:	8b c8                                           	mov    ecx,eax
   140001117:	e8 44 12 00 00                                  	call   0x140002360
   14000111c:	e8 3b 08 00 00                                  	call   0x14000195c
   140001121:	8b d8                                           	mov    ebx,eax
   140001123:	e8 f8 10 00 00                                  	call   0x140002220
   140001128:	b9 01 00 00 00                                  	mov    ecx,0x1
   14000112d:	89 18                                           	mov    DWORD PTR [rax],ebx
   14000112f:	e8 c4 04 00 00                                  	call   0x1400015f8
   140001134:	84 c0                                           	test   al,al
   140001136:	74 73                                           	je     0x1400011ab
   140001138:	e8 bb 09 00 00                                  	call   0x140001af8
   14000113d:	48 8d 0d f0 09 00 00                            	lea    rcx,[rip+0x9f0]        # 0x140001b34
   140001144:	e8 cf 02 00 00                                  	call   0x140001418
   140001149:	e8 06 08 00 00                                  	call   0x140001954
   14000114e:	8b c8                                           	mov    ecx,eax
   140001150:	e8 1b 11 00 00                                  	call   0x140002270
   140001155:	85 c0                                           	test   eax,eax
   140001157:	75 52                                           	jne    0x1400011ab
   140001159:	e8 12 08 00 00                                  	call   0x140001970
   14000115e:	e8 59 08 00 00                                  	call   0x1400019bc
   140001163:	85 c0                                           	test   eax,eax
   140001165:	74 0c                                           	je     0x140001173
   140001167:	48 8d 0d e2 07 00 00                            	lea    rcx,[rip+0x7e2]        # 0x140001950
   14000116e:	e8 bd 10 00 00                                  	call   0x140002230
   140001173:	e8 18 08 00 00                                  	call   0x140001990
   140001178:	e8 17 08 00 00                                  	call   0x140001994
   14000117d:	e8 ea 07 00 00                                  	call   0x14000196c
   140001182:	8b c8                                           	mov    ecx,eax
   140001184:	e8 d7 10 00 00                                  	call   0x140002260
   140001189:	e8 fe 07 00 00                                  	call   0x14000198c
   14000118e:	84 c0                                           	test   al,al
   140001190:	74 05                                           	je     0x140001197
   140001192:	e8 39 11 00 00                                  	call   0x1400022d0
   140001197:	e8 d8 08 00 00                                  	call   0x140001a74
   14000119c:	e8 d7 08 00 00                                  	call   0x140001a78
   1400011a1:	85 c0                                           	test   eax,eax
   1400011a3:	75 06                                           	jne    0x1400011ab
   1400011a5:	48 83 c4 20                                     	add    rsp,0x20
   1400011a9:	5b                                              	pop    rbx
   1400011aa:	c3                                              	ret
   1400011ab:	b9 07 00 00 00                                  	mov    ecx,0x7
   1400011b0:	e8 db 08 00 00                                  	call   0x140001a90
   1400011b5:	cc                                              	int3
   1400011b6:	cc                                              	int3
   1400011b7:	cc                                              	int3
   1400011b8:	48 83 ec 28                                     	sub    rsp,0x28
   1400011bc:	e8 d7 07 00 00                                  	call   0x140001998
   1400011c1:	33 c0                                           	xor    eax,eax
   1400011c3:	48 83 c4 28                                     	add    rsp,0x28
   1400011c7:	c3                                              	ret
   1400011c8:	48 83 ec 28                                     	sub    rsp,0x28
   1400011cc:	e8 af 08 00 00                                  	call   0x140001a80
   1400011d1:	e8 92 07 00 00                                  	call   0x140001968
   1400011d6:	8b c8                                           	mov    ecx,eax
   1400011d8:	48 83 c4 28                                     	add    rsp,0x28
   1400011dc:	e9 8f 11 00 00                                  	jmp    0x140002370
   1400011e1:	cc                                              	int3
   1400011e2:	cc                                              	int3
   1400011e3:	cc                                              	int3
   1400011e4:	48 83 ec 28                                     	sub    rsp,0x28
   1400011e8:	e8 67 07 00 00                                  	call   0x140001954
   1400011ed:	8b c8                                           	mov    ecx,eax
   1400011ef:	48 83 c4 28                                     	add    rsp,0x28
   1400011f3:	e9 78 10 00 00                                  	jmp    0x140002270
   1400011f8:	e9 d3 10 00 00                                  	jmp    0x1400022d0
   1400011fd:	cc                                              	int3
   1400011fe:	cc                                              	int3
   1400011ff:	cc                                              	int3
   140001200:	b9 01 00 00 00                                  	mov    ecx,0x1
   140001205:	e9 46 11 00 00                                  	jmp    0x140002350
   14000120a:	cc                                              	int3
   14000120b:	cc                                              	int3
   14000120c:	48 83 ec 28                                     	sub    rsp,0x28
   140001210:	e8 4b 07 00 00                                  	call   0x140001960
   140001215:	8b c8                                           	mov    ecx,eax
   140001217:	48 83 c4 28                                     	add    rsp,0x28
   14000121b:	e9 40 11 00 00                                  	jmp    0x140002360
   140001220:	40 53                                           	rex push rbx
   140001222:	48 83 ec 20                                     	sub    rsp,0x20
   140001226:	e8 31 07 00 00                                  	call   0x14000195c
   14000122b:	8b d8                                           	mov    ebx,eax
   14000122d:	e8 ee 0f 00 00                                  	call   0x140002220
   140001232:	89 18                                           	mov    DWORD PTR [rax],ebx
   140001234:	48 83 c4 20                                     	add    rsp,0x20
   140001238:	5b                                              	pop    rbx
   140001239:	c3                                              	ret
   14000123a:	cc                                              	int3
   14000123b:	cc                                              	int3
   14000123c:	48 89 5c 24 08                                  	mov    QWORD PTR [rsp+0x8],rbx
   140001241:	57                                              	push   rdi
   140001242:	48 83 ec 20                                     	sub    rsp,0x20
   140001246:	e8 75 10 00 00                                  	call   0x1400022c0
   14000124b:	48 8b f8                                        	mov    rdi,rax
   14000124e:	e8 bd 0f 00 00                                  	call   0x140002210
   140001253:	48 8b 18                                        	mov    rbx,QWORD PTR [rax]
   140001256:	e8 a5 0f 00 00                                  	call   0x140002200
   14000125b:	4c 8b c7                                        	mov    r8,rdi
   14000125e:	48 8b d3                                        	mov    rdx,rbx
   140001261:	8b 08                                           	mov    ecx,DWORD PTR [rax]
   140001263:	48 8b 5c 24 30                                  	mov    rbx,QWORD PTR [rsp+0x30]
   140001268:	48 83 c4 20                                     	add    rsp,0x20
   14000126c:	5f                                              	pop    rdi
   14000126d:	e9 94 fd ff ff                                  	jmp    0x140001006
   140001272:	cc                                              	int3
   140001273:	cc                                              	int3
   140001274:	48 89 5c 24 08                                  	mov    QWORD PTR [rsp+0x8],rbx
   140001279:	48 89 74 24 10                                  	mov    QWORD PTR [rsp+0x10],rsi
   14000127e:	57                                              	push   rdi
   14000127f:	48 83 ec 30                                     	sub    rsp,0x30
   140001283:	b9 01 00 00 00                                  	mov    ecx,0x1
   140001288:	e8 03 03 00 00                                  	call   0x140001590
   14000128d:	84 c0                                           	test   al,al
   14000128f:	0f 84 36 01 00 00                               	je     0x1400013cb
   140001295:	40 32 f6                                        	xor    sil,sil
   140001298:	40 88 74 24 20                                  	mov    BYTE PTR [rsp+0x20],sil
   14000129d:	e8 8e 02 00 00                                  	call   0x140001530
   1400012a2:	8a d8                                           	mov    bl,al
   1400012a4:	8b 0d d6 2d 00 00                               	mov    ecx,DWORD PTR [rip+0x2dd6]        # 0x140004080
   1400012aa:	83 f9 01                                        	cmp    ecx,0x1
   1400012ad:	0f 84 23 01 00 00                               	je     0x1400013d6
   1400012b3:	85 c9                                           	test   ecx,ecx
   1400012b5:	75 4a                                           	jne    0x140001301
   1400012b7:	c7 05 bf 2d 00 00 01 00 00 00                   	mov    DWORD PTR [rip+0x2dbf],0x1        # 0x140004080
   1400012c1:	48 8d 15 a0 20 00 00                            	lea    rdx,[rip+0x20a0]        # 0x140003368
   1400012c8:	48 8d 0d 81 20 00 00                            	lea    rcx,[rip+0x2081]        # 0x140003350
   1400012cf:	e8 2c 10 00 00                                  	call   0x140002300
   1400012d4:	85 c0                                           	test   eax,eax
   1400012d6:	74 0a                                           	je     0x1400012e2
   1400012d8:	b8 ff 00 00 00                                  	mov    eax,0xff
   1400012dd:	e9 d9 00 00 00                                  	jmp    0x1400013bb
   1400012e2:	48 8d 15 5f 20 00 00                            	lea    rdx,[rip+0x205f]        # 0x140003348
   1400012e9:	48 8d 0d 48 20 00 00                            	lea    rcx,[rip+0x2048]        # 0x140003338
   1400012f0:	e8 fb 0f 00 00                                  	call   0x1400022f0
   1400012f5:	c7 05 81 2d 00 00 02 00 00 00                   	mov    DWORD PTR [rip+0x2d81],0x2        # 0x140004080
   1400012ff:	eb 08                                           	jmp    0x140001309
   140001301:	40 b6 01                                        	mov    sil,0x1
   140001304:	40 88 74 24 20                                  	mov    BYTE PTR [rsp+0x20],sil
   140001309:	8a cb                                           	mov    cl,bl
   14000130b:	e8 5c 02 00 00                                  	call   0x14000156c
   140001310:	e8 b3 06 00 00                                  	call   0x1400019c8
   140001315:	48 8b d8                                        	mov    rbx,rax
   140001318:	48 83 38 00                                     	cmp    QWORD PTR [rax],0x0
   14000131c:	74 1e                                           	je     0x14000133c
   14000131e:	48 8b c8                                        	mov    rcx,rax
   140001321:	e8 72 01 00 00                                  	call   0x140001498
   140001326:	84 c0                                           	test   al,al
   140001328:	74 12                                           	je     0x14000133c
   14000132a:	45 33 c0                                        	xor    r8d,r8d
   14000132d:	41 8d 50 02                                     	lea    edx,[r8+0x2]
   140001331:	33 c9                                           	xor    ecx,ecx
   140001333:	48 8b 03                                        	mov    rax,QWORD PTR [rbx]
   140001336:	ff 15 d4 1f 00 00                               	call   QWORD PTR [rip+0x1fd4]        # 0x140003310
   14000133c:	e8 8f 06 00 00                                  	call   0x1400019d0
   140001341:	48 8b d8                                        	mov    rbx,rax
   140001344:	48 83 38 00                                     	cmp    QWORD PTR [rax],0x0
   140001348:	74 14                                           	je     0x14000135e
   14000134a:	48 8b c8                                        	mov    rcx,rax
   14000134d:	e8 46 01 00 00                                  	call   0x140001498
   140001352:	84 c0                                           	test   al,al
   140001354:	74 08                                           	je     0x14000135e
   140001356:	48 8b 0b                                        	mov    rcx,QWORD PTR [rbx]
   140001359:	e8 c2 0f 00 00                                  	call   0x140002320
   14000135e:	e8 5d 0f 00 00                                  	call   0x1400022c0
   140001363:	48 8b f8                                        	mov    rdi,rax
   140001366:	e8 a5 0e 00 00                                  	call   0x140002210
   14000136b:	48 8b 18                                        	mov    rbx,QWORD PTR [rax]
   14000136e:	e8 8d 0e 00 00                                  	call   0x140002200
   140001373:	4c 8b c7                                        	mov    r8,rdi
   140001376:	48 8b d3                                        	mov    rdx,rbx
   140001379:	8b 08                                           	mov    ecx,DWORD PTR [rax]
   14000137b:	e8 86 fc ff ff                                  	call   0x140001006
   140001380:	8b d8                                           	mov    ebx,eax
   140001382:	e8 99 06 00 00                                  	call   0x140001a20
   140001387:	84 c0                                           	test   al,al
   140001389:	74 55                                           	je     0x1400013e0
   14000138b:	40 84 f6                                        	test   sil,sil
   14000138e:	75 05                                           	jne    0x140001395
   140001390:	e8 bb 0e 00 00                                  	call   0x140002250
   140001395:	33 d2                                           	xor    edx,edx
   140001397:	b1 01                                           	mov    cl,0x1
   140001399:	e8 2e 02 00 00                                  	call   0x1400015cc
   14000139e:	8b c3                                           	mov    eax,ebx
   1400013a0:	eb 19                                           	jmp    0x1400013bb
   1400013a2:	8b d8                                           	mov    ebx,eax
   1400013a4:	e8 77 06 00 00                                  	call   0x140001a20
   1400013a9:	84 c0                                           	test   al,al
   1400013ab:	74 3b                                           	je     0x1400013e8
   1400013ad:	80 7c 24 20 00                                  	cmp    BYTE PTR [rsp+0x20],0x0
   1400013b2:	75 05                                           	jne    0x1400013b9
   1400013b4:	e8 87 0e 00 00                                  	call   0x140002240
   1400013b9:	8b c3                                           	mov    eax,ebx
   1400013bb:	48 8b 5c 24 40                                  	mov    rbx,QWORD PTR [rsp+0x40]
   1400013c0:	48 8b 74 24 48                                  	mov    rsi,QWORD PTR [rsp+0x48]
   1400013c5:	48 83 c4 30                                     	add    rsp,0x30
   1400013c9:	5f                                              	pop    rdi
   1400013ca:	c3                                              	ret
   1400013cb:	b9 07 00 00 00                                  	mov    ecx,0x7
   1400013d0:	e8 bb 06 00 00                                  	call   0x140001a90
   1400013d5:	90                                              	nop
   1400013d6:	b9 07 00 00 00                                  	mov    ecx,0x7
   1400013db:	e8 b0 06 00 00                                  	call   0x140001a90
   1400013e0:	8b cb                                           	mov    ecx,ebx
   1400013e2:	e8 99 0f 00 00                                  	call   0x140002380
   1400013e7:	90                                              	nop
   1400013e8:	8b cb                                           	mov    ecx,ebx
   1400013ea:	e8 c1 0e 00 00                                  	call   0x1400022b0
   1400013ef:	90                                              	nop
   1400013f0:	48 83 ec 28                                     	sub    rsp,0x28
   1400013f4:	e8 37 04 00 00                                  	call   0x140001830
   1400013f9:	48 83 c4 28                                     	add    rsp,0x28
   1400013fd:	e9 72 fe ff ff                                  	jmp    0x140001274
   140001402:	cc                                              	int3
   140001403:	cc                                              	int3
   140001404:	48 83 ec 28                                     	sub    rsp,0x28
   140001408:	e8 23 04 00 00                                  	call   0x140001830
   14000140d:	48 83 c4 28                                     	add    rsp,0x28
   140001411:	e9 5e fe ff ff                                  	jmp    0x140001274
   140001416:	cc                                              	int3
   140001417:	cc                                              	int3
   140001418:	48 83 ec 28                                     	sub    rsp,0x28
   14000141c:	e8 0f 00 00 00                                  	call   0x140001430
   140001421:	48 f7 d8                                        	neg    rax
   140001424:	1b c0                                           	sbb    eax,eax
   140001426:	f7 d8                                           	neg    eax
   140001428:	ff c8                                           	dec    eax
   14000142a:	48 83 c4 28                                     	add    rsp,0x28
   14000142e:	c3                                              	ret
   14000142f:	cc                                              	int3
   140001430:	40 53                                           	rex push rbx
   140001432:	48 83 ec 20                                     	sub    rsp,0x20
   140001436:	48 83 3d 5a 2c 00 00 ff                         	cmp    QWORD PTR [rip+0x2c5a],0xffffffffffffffff        # 0x140004098
   14000143e:	48 8b d9                                        	mov    rbx,rcx
   140001441:	75 07                                           	jne    0x14000144a
   140001443:	e8 48 0e 00 00                                  	call   0x140002290
   140001448:	eb 0f                                           	jmp    0x140001459
   14000144a:	48 8b d3                                        	mov    rdx,rbx
   14000144d:	48 8d 0d 44 2c 00 00                            	lea    rcx,[rip+0x2c44]        # 0x140004098
   140001454:	e8 b7 0e 00 00                                  	call   0x140002310
   140001459:	33 d2                                           	xor    edx,edx
   14000145b:	85 c0                                           	test   eax,eax
   14000145d:	48 0f 44 d3                                     	cmove  rdx,rbx
   140001461:	48 8b c2                                        	mov    rax,rdx
   140001464:	48 83 c4 20                                     	add    rsp,0x20
   140001468:	5b                                              	pop    rbx
   140001469:	c3                                              	ret
   14000146a:	cc                                              	int3
   14000146b:	cc                                              	int3
   14000146c:	48 83 3d 3c 2c 00 00 ff                         	cmp    QWORD PTR [rip+0x2c3c],0xffffffffffffffff        # 0x1400040b0
   140001474:	0f 84 06 0e 00 00                               	je     0x140002280
   14000147a:	48 8b d1                                        	mov    rdx,rcx
   14000147d:	48 8d 0d 2c 2c 00 00                            	lea    rcx,[rip+0x2c2c]        # 0x1400040b0
   140001484:	e9 87 0e 00 00                                  	jmp    0x140002310
   140001489:	cc                                              	int3
   14000148a:	cc                                              	int3
   14000148b:	cc                                              	int3
   14000148c:	65 48 8b 04 25 30 00 00 00                      	mov    rax,QWORD PTR gs:0x30
   140001495:	c3                                              	ret
   140001496:	cc                                              	int3
   140001497:	cc                                              	int3
   140001498:	48 83 ec 18                                     	sub    rsp,0x18
   14000149c:	4c 8b c1                                        	mov    r8,rcx
   14000149f:	b8 4d 5a 00 00                                  	mov    eax,0x5a4d
   1400014a4:	66 39 05 55 eb ff ff                            	cmp    WORD PTR [rip+0xffffffffffffeb55],ax        # 0x140000000
   1400014ab:	75 78                                           	jne    0x140001525
   1400014ad:	48 63 0d 88 eb ff ff                            	movsxd rcx,DWORD PTR [rip+0xffffffffffffeb88]        # 0x14000003c
   1400014b4:	48 8d 15 45 eb ff ff                            	lea    rdx,[rip+0xffffffffffffeb45]        # 0x140000000
   1400014bb:	48 03 ca                                        	add    rcx,rdx
   1400014be:	81 39 50 45 00 00                               	cmp    DWORD PTR [rcx],0x4550
   1400014c4:	75 5f                                           	jne    0x140001525
   1400014c6:	b8 0b 02 00 00                                  	mov    eax,0x20b
   1400014cb:	66 39 41 18                                     	cmp    WORD PTR [rcx+0x18],ax
   1400014cf:	75 54                                           	jne    0x140001525
   1400014d1:	4c 2b c2                                        	sub    r8,rdx
   1400014d4:	0f b7 51 14                                     	movzx  edx,WORD PTR [rcx+0x14]
   1400014d8:	48 83 c2 18                                     	add    rdx,0x18
   1400014dc:	48 03 d1                                        	add    rdx,rcx
   1400014df:	0f b7 41 06                                     	movzx  eax,WORD PTR [rcx+0x6]
   1400014e3:	48 8d 0c 80                                     	lea    rcx,[rax+rax*4]
   1400014e7:	4c 8d 0c ca                                     	lea    r9,[rdx+rcx*8]
   1400014eb:	48 89 14 24                                     	mov    QWORD PTR [rsp],rdx
   1400014ef:	49 3b d1                                        	cmp    rdx,r9
   1400014f2:	74 18                                           	je     0x14000150c
   1400014f4:	8b 4a 0c                                        	mov    ecx,DWORD PTR [rdx+0xc]
   1400014f7:	4c 3b c1                                        	cmp    r8,rcx
   1400014fa:	72 0a                                           	jb     0x140001506
   1400014fc:	8b 42 08                                        	mov    eax,DWORD PTR [rdx+0x8]
   1400014ff:	03 c1                                           	add    eax,ecx
   140001501:	4c 3b c0                                        	cmp    r8,rax
   140001504:	72 08                                           	jb     0x14000150e
   140001506:	48 83 c2 28                                     	add    rdx,0x28
   14000150a:	eb df                                           	jmp    0x1400014eb
   14000150c:	33 d2                                           	xor    edx,edx
   14000150e:	48 85 d2                                        	test   rdx,rdx
   140001511:	75 04                                           	jne    0x140001517
   140001513:	32 c0                                           	xor    al,al
   140001515:	eb 14                                           	jmp    0x14000152b
   140001517:	83 7a 24 00                                     	cmp    DWORD PTR [rdx+0x24],0x0
   14000151b:	7d 04                                           	jge    0x140001521
   14000151d:	32 c0                                           	xor    al,al
   14000151f:	eb 0a                                           	jmp    0x14000152b
   140001521:	b0 01                                           	mov    al,0x1
   140001523:	eb 06                                           	jmp    0x14000152b
   140001525:	32 c0                                           	xor    al,al
   140001527:	eb 02                                           	jmp    0x14000152b
   140001529:	32 c0                                           	xor    al,al
   14000152b:	48 83 c4 18                                     	add    rsp,0x18
   14000152f:	c3                                              	ret
   140001530:	48 83 ec 28                                     	sub    rsp,0x28
   140001534:	e8 37 0b 00 00                                  	call   0x140002070
   140001539:	85 c0                                           	test   eax,eax
   14000153b:	74 21                                           	je     0x14000155e
   14000153d:	65 48 8b 04 25 30 00 00 00                      	mov    rax,QWORD PTR gs:0x30
   140001546:	48 8b 48 08                                     	mov    rcx,QWORD PTR [rax+0x8]
   14000154a:	eb 05                                           	jmp    0x140001551
   14000154c:	48 3b c8                                        	cmp    rcx,rax
   14000154f:	74 14                                           	je     0x140001565
   140001551:	33 c0                                           	xor    eax,eax
   140001553:	f0 48 0f b1 0d 2c 2b 00 00                      	lock cmpxchg QWORD PTR [rip+0x2b2c],rcx        # 0x140004088
   14000155c:	75 ee                                           	jne    0x14000154c
   14000155e:	32 c0                                           	xor    al,al
   140001560:	48 83 c4 28                                     	add    rsp,0x28
   140001564:	c3                                              	ret
   140001565:	b0 01                                           	mov    al,0x1
   140001567:	eb f7                                           	jmp    0x140001560
   140001569:	cc                                              	int3
   14000156a:	cc                                              	int3
   14000156b:	cc                                              	int3
   14000156c:	40 53                                           	rex push rbx
   14000156e:	48 83 ec 20                                     	sub    rsp,0x20
   140001572:	8a d9                                           	mov    bl,cl
   140001574:	e8 f7 0a 00 00                                  	call   0x140002070
   140001579:	33 d2                                           	xor    edx,edx
   14000157b:	85 c0                                           	test   eax,eax
   14000157d:	74 0b                                           	je     0x14000158a
   14000157f:	84 db                                           	test   bl,bl
   140001581:	75 07                                           	jne    0x14000158a
   140001583:	48 87 15 fe 2a 00 00                            	xchg   QWORD PTR [rip+0x2afe],rdx        # 0x140004088
   14000158a:	48 83 c4 20                                     	add    rsp,0x20
   14000158e:	5b                                              	pop    rbx
   14000158f:	c3                                              	ret
   140001590:	48 83 ec 28                                     	sub    rsp,0x28
   140001594:	85 c9                                           	test   ecx,ecx
   140001596:	75 07                                           	jne    0x14000159f
   140001598:	c6 05 f1 2a 00 00 01                            	mov    BYTE PTR [rip+0x2af1],0x1        # 0x140004090
   14000159f:	e8 30 08 00 00                                  	call   0x140001dd4
   1400015a4:	e8 d7 0a 00 00                                  	call   0x140002080
   1400015a9:	84 c0                                           	test   al,al
   1400015ab:	75 04                                           	jne    0x1400015b1
   1400015ad:	32 c0                                           	xor    al,al
   1400015af:	eb 14                                           	jmp    0x1400015c5
   1400015b1:	e8 ca 0a 00 00                                  	call   0x140002080
   1400015b6:	84 c0                                           	test   al,al
   1400015b8:	75 09                                           	jne    0x1400015c3
   1400015ba:	33 c9                                           	xor    ecx,ecx
   1400015bc:	e8 c3 0a 00 00                                  	call   0x140002084
   1400015c1:	eb ea                                           	jmp    0x1400015ad
   1400015c3:	b0 01                                           	mov    al,0x1
   1400015c5:	48 83 c4 28                                     	add    rsp,0x28
   1400015c9:	c3                                              	ret
   1400015ca:	cc                                              	int3
   1400015cb:	cc                                              	int3
   1400015cc:	40 53                                           	rex push rbx
   1400015ce:	48 83 ec 20                                     	sub    rsp,0x20
   1400015d2:	80 3d b7 2a 00 00 00                            	cmp    BYTE PTR [rip+0x2ab7],0x0        # 0x140004090
   1400015d9:	8a d9                                           	mov    bl,cl
   1400015db:	74 04                                           	je     0x1400015e1
   1400015dd:	84 d2                                           	test   dl,dl
   1400015df:	75 0c                                           	jne    0x1400015ed
   1400015e1:	e8 9e 0a 00 00                                  	call   0x140002084
   1400015e6:	8a cb                                           	mov    cl,bl
   1400015e8:	e8 97 0a 00 00                                  	call   0x140002084
   1400015ed:	b0 01                                           	mov    al,0x1
   1400015ef:	48 83 c4 20                                     	add    rsp,0x20
   1400015f3:	5b                                              	pop    rbx
   1400015f4:	c3                                              	ret
   1400015f5:	cc                                              	int3
   1400015f6:	cc                                              	int3
   1400015f7:	cc                                              	int3
   1400015f8:	40 53                                           	rex push rbx
   1400015fa:	48 83 ec 20                                     	sub    rsp,0x20
   1400015fe:	80 3d 8c 2a 00 00 00                            	cmp    BYTE PTR [rip+0x2a8c],0x0        # 0x140004091
   140001605:	8b d9                                           	mov    ebx,ecx
   140001607:	75 67                                           	jne    0x140001670
   140001609:	83 f9 01                                        	cmp    ecx,0x1
   14000160c:	77 6a                                           	ja     0x140001678
   14000160e:	e8 5d 0a 00 00                                  	call   0x140002070
   140001613:	85 c0                                           	test   eax,eax
   140001615:	74 28                                           	je     0x14000163f
   140001617:	85 db                                           	test   ebx,ebx
   140001619:	75 24                                           	jne    0x14000163f
   14000161b:	48 8d 0d 76 2a 00 00                            	lea    rcx,[rip+0x2a76]        # 0x140004098
   140001622:	e8 b9 0c 00 00                                  	call   0x1400022e0
   140001627:	85 c0                                           	test   eax,eax
   140001629:	75 10                                           	jne    0x14000163b
   14000162b:	48 8d 0d 7e 2a 00 00                            	lea    rcx,[rip+0x2a7e]        # 0x1400040b0
   140001632:	e8 a9 0c 00 00                                  	call   0x1400022e0
   140001637:	85 c0                                           	test   eax,eax
   140001639:	74 2e                                           	je     0x140001669
   14000163b:	32 c0                                           	xor    al,al
   14000163d:	eb 33                                           	jmp    0x140001672
   14000163f:	66 0f 6f 05 09 1b 00 00                         	movdqa xmm0,XMMWORD PTR [rip+0x1b09]        # 0x140003150
   140001647:	48 83 c8 ff                                     	or     rax,0xffffffffffffffff
   14000164b:	f3 0f 7f 05 45 2a 00 00                         	movdqu XMMWORD PTR [rip+0x2a45],xmm0        # 0x140004098
   140001653:	48 89 05 4e 2a 00 00                            	mov    QWORD PTR [rip+0x2a4e],rax        # 0x1400040a8
   14000165a:	f3 0f 7f 05 4e 2a 00 00                         	movdqu XMMWORD PTR [rip+0x2a4e],xmm0        # 0x1400040b0
   140001662:	48 89 05 57 2a 00 00                            	mov    QWORD PTR [rip+0x2a57],rax        # 0x1400040c0
   140001669:	c6 05 21 2a 00 00 01                            	mov    BYTE PTR [rip+0x2a21],0x1        # 0x140004091
   140001670:	b0 01                                           	mov    al,0x1
   140001672:	48 83 c4 20                                     	add    rsp,0x20
   140001676:	5b                                              	pop    rbx
   140001677:	c3                                              	ret
   140001678:	b9 05 00 00 00                                  	mov    ecx,0x5
   14000167d:	e8 0e 04 00 00                                  	call   0x140001a90
   140001682:	cc                                              	int3
   140001683:	cc                                              	int3
   140001684:	48 89 5c 24 08                                  	mov    QWORD PTR [rsp+0x8],rbx
   140001689:	48 89 6c 24 10                                  	mov    QWORD PTR [rsp+0x10],rbp
   14000168e:	48 89 74 24 18                                  	mov    QWORD PTR [rsp+0x18],rsi
   140001693:	57                                              	push   rdi
   140001694:	48 83 ec 20                                     	sub    rsp,0x20
   140001698:	49 8b f9                                        	mov    rdi,r9
   14000169b:	49 8b f0                                        	mov    rsi,r8
   14000169e:	8b da                                           	mov    ebx,edx
   1400016a0:	48 8b e9                                        	mov    rbp,rcx
   1400016a3:	e8 c8 09 00 00                                  	call   0x140002070
   1400016a8:	85 c0                                           	test   eax,eax
   1400016aa:	75 16                                           	jne    0x1400016c2
   1400016ac:	83 fb 01                                        	cmp    ebx,0x1
   1400016af:	75 11                                           	jne    0x1400016c2
   1400016b1:	4c 8b c6                                        	mov    r8,rsi
   1400016b4:	33 d2                                           	xor    edx,edx
   1400016b6:	48 8b cd                                        	mov    rcx,rbp
   1400016b9:	48 8b c7                                        	mov    rax,rdi
   1400016bc:	ff 15 4e 1c 00 00                               	call   QWORD PTR [rip+0x1c4e]        # 0x140003310
   1400016c2:	48 8b 54 24 58                                  	mov    rdx,QWORD PTR [rsp+0x58]
   1400016c7:	8b 4c 24 50                                     	mov    ecx,DWORD PTR [rsp+0x50]
   1400016cb:	48 8b 5c 24 30                                  	mov    rbx,QWORD PTR [rsp+0x30]
   1400016d0:	48 8b 6c 24 38                                  	mov    rbp,QWORD PTR [rsp+0x38]
   1400016d5:	48 8b 74 24 40                                  	mov    rsi,QWORD PTR [rsp+0x40]
   1400016da:	48 83 c4 20                                     	add    rsp,0x20
   1400016de:	5f                                              	pop    rdi
   1400016df:	e9 4c 0c 00 00                                  	jmp    0x140002330
   1400016e4:	48 83 ec 28                                     	sub    rsp,0x28
   1400016e8:	33 c9                                           	xor    ecx,ecx
   1400016ea:	e8 09 ff ff ff                                  	call   0x1400015f8
   1400016ef:	84 c0                                           	test   al,al
   1400016f1:	0f 95 c0                                        	setne  al
   1400016f4:	48 83 c4 28                                     	add    rsp,0x28
   1400016f8:	c3                                              	ret
   1400016f9:	cc                                              	int3
   1400016fa:	cc                                              	int3
   1400016fb:	cc                                              	int3
   1400016fc:	48 83 ec 28                                     	sub    rsp,0x28
   140001700:	e8 6b 09 00 00                                  	call   0x140002070
   140001705:	85 c0                                           	test   eax,eax
   140001707:	74 07                                           	je     0x140001710
   140001709:	e8 c6 06 00 00                                  	call   0x140001dd4
   14000170e:	eb 19                                           	jmp    0x140001729
   140001710:	e8 3f 02 00 00                                  	call   0x140001954
   140001715:	8b c8                                           	mov    ecx,eax
   140001717:	e8 54 0b 00 00                                  	call   0x140002270
   14000171c:	85 c0                                           	test   eax,eax
   14000171e:	74 04                                           	je     0x140001724
   140001720:	32 c0                                           	xor    al,al
   140001722:	eb 07                                           	jmp    0x14000172b
   140001724:	e8 a7 0b 00 00                                  	call   0x1400022d0
   140001729:	b0 01                                           	mov    al,0x1
   14000172b:	48 83 c4 28                                     	add    rsp,0x28
   14000172f:	c3                                              	ret
   140001730:	48 83 ec 28                                     	sub    rsp,0x28
   140001734:	e8 37 09 00 00                                  	call   0x140002070
   140001739:	85 c0                                           	test   eax,eax
   14000173b:	74 10                                           	je     0x14000174d
   14000173d:	48 8d 0d 54 29 00 00                            	lea    rcx,[rip+0x2954]        # 0x140004098
   140001744:	48 83 c4 28                                     	add    rsp,0x28
   140001748:	e9 53 0b 00 00                                  	jmp    0x1400022a0
   14000174d:	e8 42 09 00 00                                  	call   0x140002094
   140001752:	85 c0                                           	test   eax,eax
   140001754:	75 05                                           	jne    0x14000175b
   140001756:	e8 f5 0a 00 00                                  	call   0x140002250
   14000175b:	48 83 c4 28                                     	add    rsp,0x28
   14000175f:	c3                                              	ret
   140001760:	48 83 ec 28                                     	sub    rsp,0x28
   140001764:	33 c9                                           	xor    ecx,ecx
   140001766:	e8 1d 09 00 00                                  	call   0x140002088
   14000176b:	48 83 c4 28                                     	add    rsp,0x28
   14000176f:	e9 14 09 00 00                                  	jmp    0x140002088
   140001774:	48 83 ec 28                                     	sub    rsp,0x28
   140001778:	e8 0f 09 00 00                                  	call   0x14000208c
   14000177d:	84 c0                                           	test   al,al
   14000177f:	75 04                                           	jne    0x140001785
   140001781:	32 c0                                           	xor    al,al
   140001783:	eb 12                                           	jmp    0x140001797
   140001785:	e8 02 09 00 00                                  	call   0x14000208c
   14000178a:	84 c0                                           	test   al,al
   14000178c:	75 07                                           	jne    0x140001795
   14000178e:	e8 fd 08 00 00                                  	call   0x140002090
   140001793:	eb ec                                           	jmp    0x140001781
   140001795:	b0 01                                           	mov    al,0x1
   140001797:	48 83 c4 28                                     	add    rsp,0x28
   14000179b:	c3                                              	ret
   14000179c:	48 83 ec 28                                     	sub    rsp,0x28
   1400017a0:	e8 eb 08 00 00                                  	call   0x140002090
   1400017a5:	e8 e6 08 00 00                                  	call   0x140002090
   1400017aa:	b0 01                                           	mov    al,0x1
   1400017ac:	48 83 c4 28                                     	add    rsp,0x28
   1400017b0:	c3                                              	ret
   1400017b1:	cc                                              	int3
   1400017b2:	cc                                              	int3
   1400017b3:	cc                                              	int3
   1400017b4:	48 85 c9                                        	test   rcx,rcx
   1400017b7:	74 25                                           	je     0x1400017de
   1400017b9:	b8 4d 5a 00 00                                  	mov    eax,0x5a4d
   1400017be:	66 39 01                                        	cmp    WORD PTR [rcx],ax
   1400017c1:	75 1b                                           	jne    0x1400017de
   1400017c3:	48 63 41 3c                                     	movsxd rax,DWORD PTR [rcx+0x3c]
   1400017c7:	81 3c 08 50 45 00 00                            	cmp    DWORD PTR [rax+rcx*1],0x4550
   1400017ce:	75 0e                                           	jne    0x1400017de
   1400017d0:	ba 0b 02 00 00                                  	mov    edx,0x20b
   1400017d5:	66 39 54 08 18                                  	cmp    WORD PTR [rax+rcx*1+0x18],dx
   1400017da:	0f 94 c0                                        	sete   al
   1400017dd:	c3                                              	ret
   1400017de:	32 c0                                           	xor    al,al
   1400017e0:	c3                                              	ret
   1400017e1:	cc                                              	int3
   1400017e2:	cc                                              	int3
   1400017e3:	cc                                              	int3
   1400017e4:	4c 63 41 3c                                     	movsxd r8,DWORD PTR [rcx+0x3c]
   1400017e8:	41 0f b7 44 08 14                               	movzx  eax,WORD PTR [r8+rcx*1+0x14]
   1400017ee:	4d 8d 0c 08                                     	lea    r9,[r8+rcx*1]
   1400017f2:	48 83 c0 18                                     	add    rax,0x18
   1400017f6:	4c 03 c8                                        	add    r9,rax
   1400017f9:	41 0f b7 44 08 06                               	movzx  eax,WORD PTR [r8+rcx*1+0x6]
   1400017ff:	48 8d 0c 80                                     	lea    rcx,[rax+rax*4]
   140001803:	4d 8d 04 c9                                     	lea    r8,[r9+rcx*8]
   140001807:	eb 18                                           	jmp    0x140001821
   140001809:	41 8b 49 0c                                     	mov    ecx,DWORD PTR [r9+0xc]
   14000180d:	48 3b d1                                        	cmp    rdx,rcx
   140001810:	72 0b                                           	jb     0x14000181d
   140001812:	41 8b 41 08                                     	mov    eax,DWORD PTR [r9+0x8]
   140001816:	03 c1                                           	add    eax,ecx
   140001818:	48 3b d0                                        	cmp    rdx,rax
   14000181b:	72 0c                                           	jb     0x140001829
   14000181d:	49 83 c1 28                                     	add    r9,0x28
   140001821:	4d 3b c8                                        	cmp    r9,r8
   140001824:	75 e3                                           	jne    0x140001809
   140001826:	33 c0                                           	xor    eax,eax
   140001828:	c3                                              	ret
   140001829:	49 8b c1                                        	mov    rax,r9
   14000182c:	c3                                              	ret
   14000182d:	cc                                              	int3
   14000182e:	cc                                              	int3
   14000182f:	cc                                              	int3
   140001830:	48 89 5c 24 18                                  	mov    QWORD PTR [rsp+0x18],rbx
   140001835:	55                                              	push   rbp
   140001836:	48 8b ec                                        	mov    rbp,rsp
   140001839:	48 83 ec 30                                     	sub    rsp,0x30
   14000183d:	48 8b 05 bc 27 00 00                            	mov    rax,QWORD PTR [rip+0x27bc]        # 0x140004000
   140001844:	48 bb 32 a2 df 2d 99 2b 00 00                   	movabs rbx,0x2b992ddfa232
   14000184e:	48 3b c3                                        	cmp    rax,rbx
   140001851:	75 77                                           	jne    0x1400018ca
   140001853:	48 8d 4d 10                                     	lea    rcx,[rbp+0x10]
   140001857:	48 c7 45 10 00 00 00 00                         	mov    QWORD PTR [rbp+0x10],0x0
   14000185f:	ff 15 73 1d 00 00                               	call   QWORD PTR [rip+0x1d73]        # 0x1400035d8
   140001865:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140001869:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   14000186d:	ff 15 4d 1d 00 00                               	call   QWORD PTR [rip+0x1d4d]        # 0x1400035c0
   140001873:	8b c0                                           	mov    eax,eax
   140001875:	48 31 45 f0                                     	xor    QWORD PTR [rbp-0x10],rax
   140001879:	ff 15 39 1d 00 00                               	call   QWORD PTR [rip+0x1d39]        # 0x1400035b8
   14000187f:	8b c0                                           	mov    eax,eax
   140001881:	48 8d 4d 18                                     	lea    rcx,[rbp+0x18]
   140001885:	48 31 45 f0                                     	xor    QWORD PTR [rbp-0x10],rax
   140001889:	ff 15 59 1d 00 00                               	call   QWORD PTR [rip+0x1d59]        # 0x1400035e8
   14000188f:	8b 45 18                                        	mov    eax,DWORD PTR [rbp+0x18]
   140001892:	48 8d 4d f0                                     	lea    rcx,[rbp-0x10]
   140001896:	48 c1 e0 20                                     	shl    rax,0x20
   14000189a:	48 33 45 18                                     	xor    rax,QWORD PTR [rbp+0x18]
   14000189e:	48 33 45 f0                                     	xor    rax,QWORD PTR [rbp-0x10]
   1400018a2:	48 33 c1                                        	xor    rax,rcx
   1400018a5:	48 b9 ff ff ff ff ff ff 00 00                   	movabs rcx,0xffffffffffff
   1400018af:	48 23 c1                                        	and    rax,rcx
   1400018b2:	48 b9 33 a2 df 2d 99 2b 00 00                   	movabs rcx,0x2b992ddfa233
   1400018bc:	48 3b c3                                        	cmp    rax,rbx
   1400018bf:	48 0f 44 c1                                     	cmove  rax,rcx
   1400018c3:	48 89 05 36 27 00 00                            	mov    QWORD PTR [rip+0x2736],rax        # 0x140004000
   1400018ca:	48 8b 5c 24 50                                  	mov    rbx,QWORD PTR [rsp+0x50]
   1400018cf:	48 f7 d0                                        	not    rax
   1400018d2:	48 89 05 67 27 00 00                            	mov    QWORD PTR [rip+0x2767],rax        # 0x140004040
   1400018d9:	48 83 c4 30                                     	add    rsp,0x30
   1400018dd:	5d                                              	pop    rbp
   1400018de:	c3                                              	ret
   1400018df:	cc                                              	int3
   1400018e0:	40 55                                           	rex push rbp
   1400018e2:	48 8b ec                                        	mov    rbp,rsp
   1400018e5:	48 83 ec 20                                     	sub    rsp,0x20
   1400018e9:	48 8d 4d 18                                     	lea    rcx,[rbp+0x18]
   1400018ed:	48 c7 45 18 00 00 00 00                         	mov    QWORD PTR [rbp+0x18],0x0
   1400018f5:	ff 15 dd 1c 00 00                               	call   QWORD PTR [rip+0x1cdd]        # 0x1400035d8
   1400018fb:	48 8b 45 18                                     	mov    rax,QWORD PTR [rbp+0x18]
   1400018ff:	48 89 45 10                                     	mov    QWORD PTR [rbp+0x10],rax
   140001903:	ff 15 b7 1c 00 00                               	call   QWORD PTR [rip+0x1cb7]        # 0x1400035c0
   140001909:	8b c0                                           	mov    eax,eax
   14000190b:	48 31 45 10                                     	xor    QWORD PTR [rbp+0x10],rax
   14000190f:	ff 15 a3 1c 00 00                               	call   QWORD PTR [rip+0x1ca3]        # 0x1400035b8
   140001915:	8b c0                                           	mov    eax,eax
   140001917:	48 8d 4d 20                                     	lea    rcx,[rbp+0x20]
   14000191b:	48 31 45 10                                     	xor    QWORD PTR [rbp+0x10],rax
   14000191f:	ff 15 c3 1c 00 00                               	call   QWORD PTR [rip+0x1cc3]        # 0x1400035e8
   140001925:	8b 45 20                                        	mov    eax,DWORD PTR [rbp+0x20]
   140001928:	48 8d 4d 10                                     	lea    rcx,[rbp+0x10]
   14000192c:	48 c1 e0 20                                     	shl    rax,0x20
   140001930:	48 33 45 20                                     	xor    rax,QWORD PTR [rbp+0x20]
   140001934:	48 33 45 10                                     	xor    rax,QWORD PTR [rbp+0x10]
   140001938:	48 33 c1                                        	xor    rax,rcx
   14000193b:	48 b9 ff ff ff ff ff ff 00 00                   	movabs rcx,0xffffffffffff
   140001945:	48 23 c1                                        	and    rax,rcx
   140001948:	48 83 c4 20                                     	add    rsp,0x20
   14000194c:	5d                                              	pop    rbp
   14000194d:	c3                                              	ret
   14000194e:	cc                                              	int3
   14000194f:	cc                                              	int3
   140001950:	33 c0                                           	xor    eax,eax
   140001952:	c3                                              	ret
   140001953:	cc                                              	int3
   140001954:	b8 01 00 00 00                                  	mov    eax,0x1
   140001959:	c3                                              	ret
   14000195a:	cc                                              	int3
   14000195b:	cc                                              	int3
   14000195c:	33 c0                                           	xor    eax,eax
   14000195e:	c3                                              	ret
   14000195f:	cc                                              	int3
   140001960:	b8 00 40 00 00                                  	mov    eax,0x4000
   140001965:	c3                                              	ret
   140001966:	cc                                              	int3
   140001967:	cc                                              	int3
   140001968:	33 c0                                           	xor    eax,eax
   14000196a:	c3                                              	ret
   14000196b:	cc                                              	int3
   14000196c:	33 c0                                           	xor    eax,eax
   14000196e:	c3                                              	ret
   14000196f:	cc                                              	int3
   140001970:	48 8d 0d 59 27 00 00                            	lea    rcx,[rip+0x2759]        # 0x1400040d0
   140001977:	48 ff 25 62 1c 00 00                            	rex.W jmp QWORD PTR [rip+0x1c62]        # 0x1400035e0
   14000197e:	cc                                              	int3
   14000197f:	cc                                              	int3
   140001980:	48 8d 0d 49 27 00 00                            	lea    rcx,[rip+0x2749]        # 0x1400040d0
   140001987:	e9 34 08 00 00                                  	jmp    0x1400021c0
   14000198c:	b0 01                                           	mov    al,0x1
   14000198e:	c3                                              	ret
   14000198f:	cc                                              	int3
   140001990:	c2 00 00                                        	ret    0x0
   140001993:	cc                                              	int3
   140001994:	c2 00 00                                        	ret    0x0
   140001997:	cc                                              	int3
   140001998:	48 83 ec 28                                     	sub    rsp,0x28
   14000199c:	e8 03 f7 ff ff                                  	call   0x1400010a4
   1400019a1:	48 83 08 24                                     	or     QWORD PTR [rax],0x24
   1400019a5:	e8 0a 00 00 00                                  	call   0x1400019b4
   1400019aa:	48 83 08 02                                     	or     QWORD PTR [rax],0x2
   1400019ae:	48 83 c4 28                                     	add    rsp,0x28
   1400019b2:	c3                                              	ret
   1400019b3:	cc                                              	int3
   1400019b4:	48 8d 05 25 27 00 00                            	lea    rax,[rip+0x2725]        # 0x1400040e0
   1400019bb:	c3                                              	ret
   1400019bc:	33 c0                                           	xor    eax,eax
   1400019be:	39 05 88 26 00 00                               	cmp    DWORD PTR [rip+0x2688],eax        # 0x14000404c
   1400019c4:	0f 94 c0                                        	sete   al
   1400019c7:	c3                                              	ret
   1400019c8:	48 8d 05 19 27 00 00                            	lea    rax,[rip+0x2719]        # 0x1400040e8
   1400019cf:	c3                                              	ret
   1400019d0:	48 8d 05 19 27 00 00                            	lea    rax,[rip+0x2719]        # 0x1400040f0
   1400019d7:	c3                                              	ret
   1400019d8:	c7 05 16 27 00 00 00 00 00 00                   	mov    DWORD PTR [rip+0x2716],0x0        # 0x1400040f8
   1400019e2:	c3                                              	ret
   1400019e3:	cc                                              	int3
   1400019e4:	48 81 ec 98 00 00 00                            	sub    rsp,0x98
   1400019eb:	33 d2                                           	xor    edx,edx
   1400019ed:	48 8d 4c 24 20                                  	lea    rcx,[rsp+0x20]
   1400019f2:	44 8d 42 68                                     	lea    r8d,[rdx+0x68]
   1400019f6:	e8 d5 07 00 00                                  	call   0x1400021d0
   1400019fb:	48 8d 4c 24 20                                  	lea    rcx,[rsp+0x20]
   140001a00:	ff 15 ca 1b 00 00                               	call   QWORD PTR [rip+0x1bca]        # 0x1400035d0
   140001a06:	f6 44 24 5c 01                                  	test   BYTE PTR [rsp+0x5c],0x1
   140001a0b:	b8 0a 00 00 00                                  	mov    eax,0xa
   140001a10:	66 0f 45 44 24 60                               	cmovne ax,WORD PTR [rsp+0x60]
   140001a16:	48 81 c4 98 00 00 00                            	add    rsp,0x98
   140001a1d:	c3                                              	ret
   140001a1e:	cc                                              	int3
   140001a1f:	cc                                              	int3
   140001a20:	48 83 ec 28                                     	sub    rsp,0x28
   140001a24:	33 c9                                           	xor    ecx,ecx
   140001a26:	ff 15 9c 1b 00 00                               	call   QWORD PTR [rip+0x1b9c]        # 0x1400035c8
   140001a2c:	48 85 c0                                        	test   rax,rax
   140001a2f:	74 3a                                           	je     0x140001a6b
   140001a31:	b9 4d 5a 00 00                                  	mov    ecx,0x5a4d
   140001a36:	66 39 08                                        	cmp    WORD PTR [rax],cx
   140001a39:	75 30                                           	jne    0x140001a6b
   140001a3b:	48 63 48 3c                                     	movsxd rcx,DWORD PTR [rax+0x3c]
   140001a3f:	81 3c 01 50 45 00 00                            	cmp    DWORD PTR [rcx+rax*1],0x4550
   140001a46:	75 23                                           	jne    0x140001a6b
   140001a48:	ba 0b 02 00 00                                  	mov    edx,0x20b
   140001a4d:	66 39 54 01 18                                  	cmp    WORD PTR [rcx+rax*1+0x18],dx
   140001a52:	75 17                                           	jne    0x140001a6b
   140001a54:	83 bc 01 84 00 00 00 0e                         	cmp    DWORD PTR [rcx+rax*1+0x84],0xe
   140001a5c:	76 0d                                           	jbe    0x140001a6b
   140001a5e:	83 bc 01 f8 00 00 00 00                         	cmp    DWORD PTR [rcx+rax*1+0xf8],0x0
   140001a66:	0f 95 c0                                        	setne  al
   140001a69:	eb 02                                           	jmp    0x140001a6d
   140001a6b:	32 c0                                           	xor    al,al
   140001a6d:	48 83 c4 28                                     	add    rsp,0x28
   140001a71:	c3                                              	ret
   140001a72:	cc                                              	int3
   140001a73:	cc                                              	int3
   140001a74:	33 c0                                           	xor    eax,eax
   140001a76:	c3                                              	ret
   140001a77:	cc                                              	int3
   140001a78:	e9 1b 00 00 00                                  	jmp    0x140001a98
   140001a7d:	cc                                              	int3
   140001a7e:	cc                                              	int3
   140001a7f:	cc                                              	int3
   140001a80:	48 8d 0d 15 00 00 00                            	lea    rcx,[rip+0x15]        # 0x140001a9c
   140001a87:	48 ff 25 62 1b 00 00                            	rex.W jmp QWORD PTR [rip+0x1b62]        # 0x1400035f0
   140001a8e:	cc                                              	int3
   140001a8f:	cc                                              	int3
   140001a90:	8b c9                                           	mov    ecx,ecx
   140001a92:	cd 29                                           	int    0x29
   140001a94:	c3                                              	ret
   140001a95:	cc                                              	int3
   140001a96:	cc                                              	int3
   140001a97:	cc                                              	int3
   140001a98:	33 c0                                           	xor    eax,eax
   140001a9a:	c3                                              	ret
   140001a9b:	cc                                              	int3
   140001a9c:	48 89 5c 24 08                                  	mov    QWORD PTR [rsp+0x8],rbx
   140001aa1:	57                                              	push   rdi
   140001aa2:	48 83 ec 20                                     	sub    rsp,0x20
   140001aa6:	48 8b 19                                        	mov    rbx,QWORD PTR [rcx]
   140001aa9:	48 8b f9                                        	mov    rdi,rcx
   140001aac:	81 3b 63 73 6d e0                               	cmp    DWORD PTR [rbx],0xe06d7363
   140001ab2:	75 1c                                           	jne    0x140001ad0
   140001ab4:	83 7b 18 04                                     	cmp    DWORD PTR [rbx+0x18],0x4
   140001ab8:	75 16                                           	jne    0x140001ad0
   140001aba:	8b 53 20                                        	mov    edx,DWORD PTR [rbx+0x20]
   140001abd:	8d 82 e0 fa 6c e6                               	lea    eax,[rdx-0x19930520]
   140001ac3:	83 f8 02                                        	cmp    eax,0x2
   140001ac6:	76 15                                           	jbe    0x140001add
   140001ac8:	81 fa 00 40 99 01                               	cmp    edx,0x1994000
   140001ace:	74 0d                                           	je     0x140001add
   140001ad0:	48 8b 5c 24 30                                  	mov    rbx,QWORD PTR [rsp+0x30]
   140001ad5:	33 c0                                           	xor    eax,eax
   140001ad7:	48 83 c4 20                                     	add    rsp,0x20
   140001adb:	5f                                              	pop    rdi
   140001adc:	c3                                              	ret
   140001add:	e8 be 06 00 00                                  	call   0x1400021a0
   140001ae2:	48 89 18                                        	mov    QWORD PTR [rax],rbx
   140001ae5:	48 8b 5f 08                                     	mov    rbx,QWORD PTR [rdi+0x8]
   140001ae9:	e8 c2 06 00 00                                  	call   0x1400021b0
   140001aee:	48 89 18                                        	mov    QWORD PTR [rax],rbx
   140001af1:	e8 9a 08 00 00                                  	call   0x140002390
   140001af6:	cc                                              	int3
   140001af7:	cc                                              	int3
   140001af8:	48 89 5c 24 08                                  	mov    QWORD PTR [rsp+0x8],rbx
   140001afd:	57                                              	push   rdi
   140001afe:	48 83 ec 20                                     	sub    rsp,0x20
   140001b02:	48 8d 1d 57 20 00 00                            	lea    rbx,[rip+0x2057]        # 0x140003b60
   140001b09:	48 8d 3d 50 20 00 00                            	lea    rdi,[rip+0x2050]        # 0x140003b60
   140001b10:	eb 12                                           	jmp    0x140001b24
   140001b12:	48 8b 03                                        	mov    rax,QWORD PTR [rbx]
   140001b15:	48 85 c0                                        	test   rax,rax
   140001b18:	74 06                                           	je     0x140001b20
   140001b1a:	ff 15 f0 17 00 00                               	call   QWORD PTR [rip+0x17f0]        # 0x140003310
   140001b20:	48 83 c3 08                                     	add    rbx,0x8
   140001b24:	48 3b df                                        	cmp    rbx,rdi
   140001b27:	72 e9                                           	jb     0x140001b12
   140001b29:	48 8b 5c 24 30                                  	mov    rbx,QWORD PTR [rsp+0x30]
   140001b2e:	48 83 c4 20                                     	add    rsp,0x20
   140001b32:	5f                                              	pop    rdi
   140001b33:	c3                                              	ret
   140001b34:	48 89 5c 24 08                                  	mov    QWORD PTR [rsp+0x8],rbx
   140001b39:	57                                              	push   rdi
   140001b3a:	48 83 ec 20                                     	sub    rsp,0x20
   140001b3e:	48 8d 1d 2b 20 00 00                            	lea    rbx,[rip+0x202b]        # 0x140003b70
   140001b45:	48 8d 3d 24 20 00 00                            	lea    rdi,[rip+0x2024]        # 0x140003b70
   140001b4c:	eb 12                                           	jmp    0x140001b60
   140001b4e:	48 8b 03                                        	mov    rax,QWORD PTR [rbx]
   140001b51:	48 85 c0                                        	test   rax,rax
   140001b54:	74 06                                           	je     0x140001b5c
   140001b56:	ff 15 b4 17 00 00                               	call   QWORD PTR [rip+0x17b4]        # 0x140003310
   140001b5c:	48 83 c3 08                                     	add    rbx,0x8
   140001b60:	48 3b df                                        	cmp    rbx,rdi
   140001b63:	72 e9                                           	jb     0x140001b4e
   140001b65:	48 8b 5c 24 30                                  	mov    rbx,QWORD PTR [rsp+0x30]
   140001b6a:	48 83 c4 20                                     	add    rsp,0x20
   140001b6e:	5f                                              	pop    rdi
   140001b6f:	c3                                              	ret
   140001b70:	c2 00 00                                        	ret    0x0
   140001b73:	cc                                              	int3
   140001b74:	48 8b 01                                        	mov    rax,QWORD PTR [rcx]
   140001b77:	c3                                              	ret
   140001b78:	48 8b 01                                        	mov    rax,QWORD PTR [rcx]
   140001b7b:	c3                                              	ret
   140001b7c:	48 87 0d 7d 25 00 00                            	xchg   QWORD PTR [rip+0x257d],rcx        # 0x140004100
   140001b83:	48 8b c1                                        	mov    rax,rcx
   140001b86:	c3                                              	ret
   140001b87:	cc                                              	int3
   140001b88:	48 8b 0d 71 17 00 00                            	mov    rcx,QWORD PTR [rip+0x1771]        # 0x140003300
   140001b8f:	48 8d 15 da ff ff ff                            	lea    rdx,[rip+0xffffffffffffffda]        # 0x140001b70
   140001b96:	33 c0                                           	xor    eax,eax
   140001b98:	48 3b ca                                        	cmp    rcx,rdx
   140001b9b:	0f 95 c0                                        	setne  al
   140001b9e:	c3                                              	ret
   140001b9f:	cc                                              	int3
   140001ba0:	33 c0                                           	xor    eax,eax
   140001ba2:	c3                                              	ret
   140001ba3:	cc                                              	int3
   140001ba4:	48 8d 05 55 16 00 00                            	lea    rax,[rip+0x1655]        # 0x140003200
   140001bab:	48 2b c8                                        	sub    rcx,rax
   140001bae:	48 8d 15 4b 16 00 00                            	lea    rdx,[rip+0x164b]        # 0x140003200
   140001bb5:	48 2b d0                                        	sub    rdx,rax
   140001bb8:	33 c0                                           	xor    eax,eax
   140001bba:	48 3b ca                                        	cmp    rcx,rdx
   140001bbd:	0f 96 c0                                        	setbe  al
   140001bc0:	c3                                              	ret
   140001bc1:	cc                                              	int3
   140001bc2:	cc                                              	int3
   140001bc3:	cc                                              	int3
   140001bc4:	48 83 ec 28                                     	sub    rsp,0x28
   140001bc8:	48 8b 05 59 17 00 00                            	mov    rax,QWORD PTR [rip+0x1759]        # 0x140003328
   140001bcf:	48 85 c0                                        	test   rax,rax
   140001bd2:	74 02                                           	je     0x140001bd6
   140001bd4:	ff d0                                           	call   rax
   140001bd6:	48 83 c4 28                                     	add    rsp,0x28
   140001bda:	c3                                              	ret
   140001bdb:	cc                                              	int3
   140001bdc:	48 83 ec 28                                     	sub    rsp,0x28
   140001be0:	48 8b 05 19 25 00 00                            	mov    rax,QWORD PTR [rip+0x2519]        # 0x140004100
   140001be7:	48 85 c0                                        	test   rax,rax
   140001bea:	74 06                                           	je     0x140001bf2
   140001bec:	ff 15 1e 17 00 00                               	call   QWORD PTR [rip+0x171e]        # 0x140003310
   140001bf2:	48 83 c4 28                                     	add    rsp,0x28
   140001bf6:	c3                                              	ret
   140001bf7:	cc                                              	int3
   140001bf8:	c2 00 00                                        	ret    0x0
   140001bfb:	cc                                              	int3
   140001bfc:	48 8d 15 fd 15 00 00                            	lea    rdx,[rip+0x15fd]        # 0x140003200
   140001c03:	48 8d 05 f6 15 00 00                            	lea    rax,[rip+0x15f6]        # 0x140003200
   140001c0a:	48 2b ca                                        	sub    rcx,rdx
   140001c0d:	48 2b c2                                        	sub    rax,rdx
   140001c10:	48 3b c8                                        	cmp    rcx,rax
   140001c13:	77 01                                           	ja     0x140001c16
   140001c15:	cc                                              	int3
   140001c16:	c3                                              	ret
   140001c17:	cc                                              	int3
   140001c18:	48 8d 15 e1 15 00 00                            	lea    rdx,[rip+0x15e1]        # 0x140003200
   140001c1f:	48 8d 05 da 15 00 00                            	lea    rax,[rip+0x15da]        # 0x140003200
   140001c26:	48 2b ca                                        	sub    rcx,rdx
   140001c29:	48 2b c2                                        	sub    rax,rdx
   140001c2c:	48 3b c8                                        	cmp    rcx,rax
   140001c2f:	77 07                                           	ja     0x140001c38
   140001c31:	b9 41 00 00 00                                  	mov    ecx,0x41
   140001c36:	cd 29                                           	int    0x29
   140001c38:	c3                                              	ret
   140001c39:	cc                                              	int3
   140001c3a:	cc                                              	int3
   140001c3b:	cc                                              	int3
   140001c3c:	48 83 ec 28                                     	sub    rsp,0x28
   140001c40:	4c 8d 05 b9 15 00 00                            	lea    r8,[rip+0x15b9]        # 0x140003200
   140001c47:	48 8b c1                                        	mov    rax,rcx
   140001c4a:	48 8d 15 af 15 00 00                            	lea    rdx,[rip+0x15af]        # 0x140003200
   140001c51:	49 2b c0                                        	sub    rax,r8
   140001c54:	49 2b d0                                        	sub    rdx,r8
   140001c57:	48 3b c2                                        	cmp    rax,rdx
   140001c5a:	77 0e                                           	ja     0x140001c6a
   140001c5c:	48 8b 05 c5 16 00 00                            	mov    rax,QWORD PTR [rip+0x16c5]        # 0x140003328
   140001c63:	48 85 c0                                        	test   rax,rax
   140001c66:	74 02                                           	je     0x140001c6a
   140001c68:	ff d0                                           	call   rax
   140001c6a:	48 83 c4 28                                     	add    rsp,0x28
   140001c6e:	c3                                              	ret
   140001c6f:	cc                                              	int3
   140001c70:	48 83 ec 28                                     	sub    rsp,0x28
   140001c74:	4c 8d 05 85 15 00 00                            	lea    r8,[rip+0x1585]        # 0x140003200
   140001c7b:	48 8b c1                                        	mov    rax,rcx
   140001c7e:	48 8d 15 7b 15 00 00                            	lea    rdx,[rip+0x157b]        # 0x140003200
   140001c85:	49 2b c0                                        	sub    rax,r8
   140001c88:	49 2b d0                                        	sub    rdx,r8
   140001c8b:	48 3b c2                                        	cmp    rax,rdx
   140001c8e:	77 12                                           	ja     0x140001ca2
   140001c90:	48 8b 05 69 24 00 00                            	mov    rax,QWORD PTR [rip+0x2469]        # 0x140004100
   140001c97:	48 85 c0                                        	test   rax,rax
   140001c9a:	74 06                                           	je     0x140001ca2
   140001c9c:	ff 15 6e 16 00 00                               	call   QWORD PTR [rip+0x166e]        # 0x140003310
   140001ca2:	48 83 c4 28                                     	add    rsp,0x28
   140001ca6:	c3                                              	ret
   140001ca7:	cc                                              	int3
   140001ca8:	48 2b ca                                        	sub    rcx,rdx
   140001cab:	48 8d 05 ce 14 00 00                            	lea    rax,[rip+0x14ce]        # 0x140003180
   140001cb2:	48 2b c8                                        	sub    rcx,rax
   140001cb5:	33 c0                                           	xor    eax,eax
   140001cb7:	48 83 c1 80                                     	add    rcx,0xffffffffffffff80
   140001cbb:	49 3b c8                                        	cmp    rcx,r8
   140001cbe:	0f 97 c0                                        	seta   al
   140001cc1:	c3                                              	ret
   140001cc2:	cc                                              	int3
   140001cc3:	cc                                              	int3
   140001cc4:	48 8b c1                                        	mov    rax,rcx
   140001cc7:	48 2b c2                                        	sub    rax,rdx
   140001cca:	48 8d 15 af 14 00 00                            	lea    rdx,[rip+0x14af]        # 0x140003180
   140001cd1:	48 2b c2                                        	sub    rax,rdx
   140001cd4:	48 83 c0 80                                     	add    rax,0xffffffffffffff80
   140001cd8:	49 3b c0                                        	cmp    rax,r8
   140001cdb:	76 1a                                           	jbe    0x140001cf7
   140001cdd:	48 8d 15 1c 15 00 00                            	lea    rdx,[rip+0x151c]        # 0x140003200
   140001ce4:	48 8d 05 15 15 00 00                            	lea    rax,[rip+0x1515]        # 0x140003200
   140001ceb:	48 2b ca                                        	sub    rcx,rdx
   140001cee:	48 2b c2                                        	sub    rax,rdx
   140001cf1:	48 3b c8                                        	cmp    rcx,rax
   140001cf4:	77 01                                           	ja     0x140001cf7
   140001cf6:	cc                                              	int3
   140001cf7:	c3                                              	ret
   140001cf8:	48 8b c1                                        	mov    rax,rcx
   140001cfb:	48 2b c2                                        	sub    rax,rdx
   140001cfe:	48 8d 15 7b 14 00 00                            	lea    rdx,[rip+0x147b]        # 0x140003180
   140001d05:	48 2b c2                                        	sub    rax,rdx
   140001d08:	48 83 c0 80                                     	add    rax,0xffffffffffffff80
   140001d0c:	49 3b c0                                        	cmp    rax,r8
   140001d0f:	76 20                                           	jbe    0x140001d31
   140001d11:	48 8d 15 e8 14 00 00                            	lea    rdx,[rip+0x14e8]        # 0x140003200
   140001d18:	48 8d 05 e1 14 00 00                            	lea    rax,[rip+0x14e1]        # 0x140003200
   140001d1f:	48 2b ca                                        	sub    rcx,rdx
   140001d22:	48 2b c2                                        	sub    rax,rdx
   140001d25:	48 3b c8                                        	cmp    rcx,rax
   140001d28:	77 07                                           	ja     0x140001d31
   140001d2a:	b9 41 00 00 00                                  	mov    ecx,0x41
   140001d2f:	cd 29                                           	int    0x29
   140001d31:	c3                                              	ret
   140001d32:	cc                                              	int3
   140001d33:	cc                                              	int3
   140001d34:	c2 00 00                                        	ret    0x0
   140001d37:	cc                                              	int3
   140001d38:	48 83 ec 28                                     	sub    rsp,0x28
   140001d3c:	48 8b c1                                        	mov    rax,rcx
   140001d3f:	48 2b c2                                        	sub    rax,rdx
   140001d42:	48 8d 15 37 14 00 00                            	lea    rdx,[rip+0x1437]        # 0x140003180
   140001d49:	48 2b c2                                        	sub    rax,rdx
   140001d4c:	48 83 c0 80                                     	add    rax,0xffffffffffffff80
   140001d50:	49 3b c0                                        	cmp    rax,r8
   140001d53:	76 2a                                           	jbe    0x140001d7f
   140001d55:	4c 8d 05 a4 14 00 00                            	lea    r8,[rip+0x14a4]        # 0x140003200
   140001d5c:	48 8b d1                                        	mov    rdx,rcx
   140001d5f:	48 8d 05 9a 14 00 00                            	lea    rax,[rip+0x149a]        # 0x140003200
   140001d66:	49 2b d0                                        	sub    rdx,r8
   140001d69:	49 2b c0                                        	sub    rax,r8
   140001d6c:	48 3b d0                                        	cmp    rdx,rax
   140001d6f:	77 0e                                           	ja     0x140001d7f
   140001d71:	48 8b 05 b0 15 00 00                            	mov    rax,QWORD PTR [rip+0x15b0]        # 0x140003328
   140001d78:	48 85 c0                                        	test   rax,rax
   140001d7b:	74 02                                           	je     0x140001d7f
   140001d7d:	ff d0                                           	call   rax
   140001d7f:	48 83 c4 28                                     	add    rsp,0x28
   140001d83:	c3                                              	ret
   140001d84:	48 83 ec 28                                     	sub    rsp,0x28
   140001d88:	48 8b c1                                        	mov    rax,rcx
   140001d8b:	48 2b c2                                        	sub    rax,rdx
   140001d8e:	48 8d 15 eb 13 00 00                            	lea    rdx,[rip+0x13eb]        # 0x140003180
   140001d95:	48 2b c2                                        	sub    rax,rdx
   140001d98:	48 83 c0 80                                     	add    rax,0xffffffffffffff80
   140001d9c:	49 3b c0                                        	cmp    rax,r8
   140001d9f:	76 2e                                           	jbe    0x140001dcf
   140001da1:	4c 8d 05 58 14 00 00                            	lea    r8,[rip+0x1458]        # 0x140003200
   140001da8:	48 8b d1                                        	mov    rdx,rcx
   140001dab:	48 8d 05 4e 14 00 00                            	lea    rax,[rip+0x144e]        # 0x140003200
   140001db2:	49 2b d0                                        	sub    rdx,r8
   140001db5:	49 2b c0                                        	sub    rax,r8
   140001db8:	48 3b d0                                        	cmp    rdx,rax
   140001dbb:	77 12                                           	ja     0x140001dcf
   140001dbd:	48 8b 05 3c 23 00 00                            	mov    rax,QWORD PTR [rip+0x233c]        # 0x140004100
   140001dc4:	48 85 c0                                        	test   rax,rax
   140001dc7:	74 06                                           	je     0x140001dcf
   140001dc9:	ff 15 41 15 00 00                               	call   QWORD PTR [rip+0x1541]        # 0x140003310
   140001dcf:	48 83 c4 28                                     	add    rsp,0x28
   140001dd3:	c3                                              	ret
   140001dd4:	48 89 5c 24 10                                  	mov    QWORD PTR [rsp+0x10],rbx
   140001dd9:	48 89 6c 24 18                                  	mov    QWORD PTR [rsp+0x18],rbp
   140001dde:	48 89 74 24 20                                  	mov    QWORD PTR [rsp+0x20],rsi
   140001de3:	57                                              	push   rdi
   140001de4:	48 83 ec 10                                     	sub    rsp,0x10
   140001de8:	33 c0                                           	xor    eax,eax
   140001dea:	33 c9                                           	xor    ecx,ecx
   140001dec:	0f a2                                           	cpuid
   140001dee:	81 f1 6e 74 65 6c                               	xor    ecx,0x6c65746e
   140001df4:	81 f2 69 6e 65 49                               	xor    edx,0x49656e69
   140001dfa:	0b d1                                           	or     edx,ecx
   140001dfc:	8b e8                                           	mov    ebp,eax
   140001dfe:	b8 01 00 00 00                                  	mov    eax,0x1
   140001e03:	81 f3 47 65 6e 75                               	xor    ebx,0x756e6547
   140001e09:	0b d3                                           	or     edx,ebx
   140001e0b:	8d 48 ff                                        	lea    ecx,[rax-0x1]
   140001e0e:	0f a2                                           	cpuid
   140001e10:	8b f9                                           	mov    edi,ecx
   140001e12:	75 5e                                           	jne    0x140001e72
   140001e14:	25 f0 3f ff 0f                                  	and    eax,0xfff3ff0
   140001e19:	48 c7 05 34 22 00 00 00 80 00 00                	mov    QWORD PTR [rip+0x2234],0x8000        # 0x140004058
   140001e24:	48 c7 05 31 22 00 00 ff ff ff ff                	mov    QWORD PTR [rip+0x2231],0xffffffffffffffff        # 0x140004060
   140001e2f:	3d c0 06 01 00                                  	cmp    eax,0x106c0
   140001e34:	74 28                                           	je     0x140001e5e
   140001e36:	3d 60 06 02 00                                  	cmp    eax,0x20660
   140001e3b:	74 21                                           	je     0x140001e5e
   140001e3d:	3d 70 06 02 00                                  	cmp    eax,0x20670
   140001e42:	74 1a                                           	je     0x140001e5e
   140001e44:	05 b0 f9 fc ff                                  	add    eax,0xfffcf9b0
   140001e49:	83 f8 20                                        	cmp    eax,0x20
   140001e4c:	77 24                                           	ja     0x140001e72
   140001e4e:	48 b9 01 00 01 00 01 00 00 00                   	movabs rcx,0x100010001
   140001e58:	48 0f a3 c1                                     	bt     rcx,rax
   140001e5c:	73 14                                           	jae    0x140001e72
   140001e5e:	44 8b 05 a3 22 00 00                            	mov    r8d,DWORD PTR [rip+0x22a3]        # 0x140004108
   140001e65:	41 83 c8 01                                     	or     r8d,0x1
   140001e69:	44 89 05 98 22 00 00                            	mov    DWORD PTR [rip+0x2298],r8d        # 0x140004108
   140001e70:	eb 07                                           	jmp    0x140001e79
   140001e72:	44 8b 05 8f 22 00 00                            	mov    r8d,DWORD PTR [rip+0x228f]        # 0x140004108
   140001e79:	45 33 c9                                        	xor    r9d,r9d
   140001e7c:	45 8b d9                                        	mov    r11d,r9d
   140001e7f:	83 fd 07                                        	cmp    ebp,0x7
   140001e82:	7c 42                                           	jl     0x140001ec6
   140001e84:	33 c9                                           	xor    ecx,ecx
   140001e86:	41 8d 41 07                                     	lea    eax,[r9+0x7]
   140001e8a:	0f a2                                           	cpuid
   140001e8c:	8b f2                                           	mov    esi,edx
   140001e8e:	44 8b d3                                        	mov    r10d,ebx
   140001e91:	0f ba e3 09                                     	bt     ebx,0x9
   140001e95:	73 0b                                           	jae    0x140001ea2
   140001e97:	41 83 c8 02                                     	or     r8d,0x2
   140001e9b:	44 89 05 66 22 00 00                            	mov    DWORD PTR [rip+0x2266],r8d        # 0x140004108
   140001ea2:	83 f8 01                                        	cmp    eax,0x1
   140001ea5:	7c 0d                                           	jl     0x140001eb4
   140001ea7:	b8 07 00 00 00                                  	mov    eax,0x7
   140001eac:	8d 48 fa                                        	lea    ecx,[rax-0x6]
   140001eaf:	0f a2                                           	cpuid
   140001eb1:	44 8b ca                                        	mov    r9d,edx
   140001eb4:	b8 24 00 00 00                                  	mov    eax,0x24
   140001eb9:	3b e8                                           	cmp    ebp,eax
   140001ebb:	7c 0f                                           	jl     0x140001ecc
   140001ebd:	33 c9                                           	xor    ecx,ecx
   140001ebf:	0f a2                                           	cpuid
   140001ec1:	44 8b db                                        	mov    r11d,ebx
   140001ec4:	eb 06                                           	jmp    0x140001ecc
   140001ec6:	45 8b d1                                        	mov    r10d,r9d
   140001ec9:	41 8b f1                                        	mov    esi,r9d
   140001ecc:	48 8b 05 95 21 00 00                            	mov    rax,QWORD PTR [rip+0x2195]        # 0x140004068
   140001ed3:	48 83 e0 fe                                     	and    rax,0xfffffffffffffffe
   140001ed7:	c7 05 6f 21 00 00 01 00 00 00                   	mov    DWORD PTR [rip+0x216f],0x1        # 0x140004050
   140001ee1:	c7 05 69 21 00 00 02 00 00 00                   	mov    DWORD PTR [rip+0x2169],0x2        # 0x140004054
   140001eeb:	48 89 05 76 21 00 00                            	mov    QWORD PTR [rip+0x2176],rax        # 0x140004068
   140001ef2:	0f ba e7 14                                     	bt     edi,0x14
   140001ef6:	73 1f                                           	jae    0x140001f17
   140001ef8:	48 83 e0 ef                                     	and    rax,0xffffffffffffffef
   140001efc:	c7 05 4a 21 00 00 02 00 00 00                   	mov    DWORD PTR [rip+0x214a],0x2        # 0x140004050
   140001f06:	48 89 05 5b 21 00 00                            	mov    QWORD PTR [rip+0x215b],rax        # 0x140004068
   140001f0d:	c7 05 3d 21 00 00 06 00 00 00                   	mov    DWORD PTR [rip+0x213d],0x6        # 0x140004054
   140001f17:	0f ba e7 1b                                     	bt     edi,0x1b
   140001f1b:	0f 83 37 01 00 00                               	jae    0x140002058
   140001f21:	33 c9                                           	xor    ecx,ecx
   140001f23:	0f 01 d0                                        	xgetbv
   140001f26:	48 c1 e2 20                                     	shl    rdx,0x20
   140001f2a:	48 0b d0                                        	or     rdx,rax
   140001f2d:	48 89 54 24 20                                  	mov    QWORD PTR [rsp+0x20],rdx
   140001f32:	0f ba e7 1c                                     	bt     edi,0x1c
   140001f36:	0f 83 00 01 00 00                               	jae    0x14000203c
   140001f3c:	48 8b 44 24 20                                  	mov    rax,QWORD PTR [rsp+0x20]
   140001f41:	24 06                                           	and    al,0x6
   140001f43:	3c 06                                           	cmp    al,0x6
   140001f45:	0f 85 f1 00 00 00                               	jne    0x14000203c
   140001f4b:	8b 05 03 21 00 00                               	mov    eax,DWORD PTR [rip+0x2103]        # 0x140004054
   140001f51:	b2 e0                                           	mov    dl,0xe0
   140001f53:	83 c8 08                                        	or     eax,0x8
   140001f56:	c7 05 f0 20 00 00 03 00 00 00                   	mov    DWORD PTR [rip+0x20f0],0x3        # 0x140004050
   140001f60:	89 05 ee 20 00 00                               	mov    DWORD PTR [rip+0x20ee],eax        # 0x140004054
   140001f66:	41 f6 c2 20                                     	test   r10b,0x20
   140001f6a:	74 62                                           	je     0x140001fce
   140001f6c:	83 c8 20                                        	or     eax,0x20
   140001f6f:	c7 05 d7 20 00 00 05 00 00 00                   	mov    DWORD PTR [rip+0x20d7],0x5        # 0x140004050
   140001f79:	89 05 d5 20 00 00                               	mov    DWORD PTR [rip+0x20d5],eax        # 0x140004054
   140001f7f:	b9 00 00 03 d0                                  	mov    ecx,0xd0030000
   140001f84:	48 8b 05 dd 20 00 00                            	mov    rax,QWORD PTR [rip+0x20dd]        # 0x140004068
   140001f8b:	44 23 d1                                        	and    r10d,ecx
   140001f8e:	48 83 e0 fd                                     	and    rax,0xfffffffffffffffd
   140001f92:	48 89 05 cf 20 00 00                            	mov    QWORD PTR [rip+0x20cf],rax        # 0x140004068
   140001f99:	44 3b d1                                        	cmp    r10d,ecx
   140001f9c:	75 37                                           	jne    0x140001fd5
   140001f9e:	48 8b 44 24 20                                  	mov    rax,QWORD PTR [rsp+0x20]
   140001fa3:	22 c2                                           	and    al,dl
   140001fa5:	3a c2                                           	cmp    al,dl
   140001fa7:	75 25                                           	jne    0x140001fce
   140001fa9:	48 8b 05 b8 20 00 00                            	mov    rax,QWORD PTR [rip+0x20b8]        # 0x140004068
   140001fb0:	83 0d 9d 20 00 00 40                            	or     DWORD PTR [rip+0x209d],0x40        # 0x140004054
   140001fb7:	48 83 e0 db                                     	and    rax,0xffffffffffffffdb
   140001fbb:	c7 05 8b 20 00 00 06 00 00 00                   	mov    DWORD PTR [rip+0x208b],0x6        # 0x140004050
   140001fc5:	48 89 05 9c 20 00 00                            	mov    QWORD PTR [rip+0x209c],rax        # 0x140004068
   140001fcc:	eb 07                                           	jmp    0x140001fd5
   140001fce:	48 8b 05 93 20 00 00                            	mov    rax,QWORD PTR [rip+0x2093]        # 0x140004068
   140001fd5:	0f ba e6 17                                     	bt     esi,0x17
   140001fd9:	73 0c                                           	jae    0x140001fe7
   140001fdb:	48 0f ba f0 18                                  	btr    rax,0x18
   140001fe0:	48 89 05 81 20 00 00                            	mov    QWORD PTR [rip+0x2081],rax        # 0x140004068
   140001fe7:	41 0f ba e1 13                                  	bt     r9d,0x13
   140001fec:	73 4e                                           	jae    0x14000203c
   140001fee:	48 8b 44 24 20                                  	mov    rax,QWORD PTR [rsp+0x20]
   140001ff3:	22 c2                                           	and    al,dl
   140001ff5:	3a c2                                           	cmp    al,dl
   140001ff7:	75 43                                           	jne    0x14000203c
   140001ff9:	48 8b 15 68 20 00 00                            	mov    rdx,QWORD PTR [rip+0x2068]        # 0x140004068
   140002000:	41 8b c3                                        	mov    eax,r11d
   140002003:	48 c1 e8 10                                     	shr    rax,0x10
   140002007:	41 8b cb                                        	mov    ecx,r11d
   14000200a:	83 e0 06                                        	and    eax,0x6
   14000200d:	81 e1 ff 00 04 00                               	and    ecx,0x400ff
   140002013:	48 0d 29 00 00 01                               	or     rax,0x1000029
   140002019:	89 0d ed 20 00 00                               	mov    DWORD PTR [rip+0x20ed],ecx        # 0x14000410c
   14000201f:	48 f7 d0                                        	not    rax
   140002022:	48 23 d0                                        	and    rdx,rax
   140002025:	48 89 15 3c 20 00 00                            	mov    QWORD PTR [rip+0x203c],rdx        # 0x140004068
   14000202c:	80 f9 01                                        	cmp    cl,0x1
   14000202f:	76 0b                                           	jbe    0x14000203c
   140002031:	48 83 e2 bf                                     	and    rdx,0xffffffffffffffbf
   140002035:	48 89 15 2c 20 00 00                            	mov    QWORD PTR [rip+0x202c],rdx        # 0x140004068
   14000203c:	41 0f ba e1 15                                  	bt     r9d,0x15
   140002041:	73 15                                           	jae    0x140002058
   140002043:	48 8b 44 24 20                                  	mov    rax,QWORD PTR [rsp+0x20]
   140002048:	48 0f ba e0 13                                  	bt     rax,0x13
   14000204d:	73 09                                           	jae    0x140002058
   14000204f:	48 0f ba 35 10 20 00 00 07                      	btr    QWORD PTR [rip+0x2010],0x7        # 0x140004068
   140002058:	48 8b 5c 24 28                                  	mov    rbx,QWORD PTR [rsp+0x28]
   14000205d:	33 c0                                           	xor    eax,eax
   14000205f:	48 8b 6c 24 30                                  	mov    rbp,QWORD PTR [rsp+0x30]
   140002064:	48 8b 74 24 38                                  	mov    rsi,QWORD PTR [rsp+0x38]
   140002069:	48 83 c4 10                                     	add    rsp,0x10
   14000206d:	5f                                              	pop    rdi
   14000206e:	c3                                              	ret
   14000206f:	cc                                              	int3
   140002070:	33 c0                                           	xor    eax,eax
   140002072:	39 05 f8 1f 00 00                               	cmp    DWORD PTR [rip+0x1ff8],eax        # 0x140004070
   140002078:	0f 95 c0                                        	setne  al
   14000207b:	c3                                              	ret
   14000207c:	cc                                              	int3
   14000207d:	cc                                              	int3
   14000207e:	cc                                              	int3
   14000207f:	cc                                              	int3
   140002080:	b0 01                                           	mov    al,0x1
   140002082:	c3                                              	ret
   140002083:	cc                                              	int3
   140002084:	b0 01                                           	mov    al,0x1
   140002086:	c3                                              	ret
   140002087:	cc                                              	int3
   140002088:	b0 01                                           	mov    al,0x1
   14000208a:	c3                                              	ret
   14000208b:	cc                                              	int3
   14000208c:	b0 01                                           	mov    al,0x1
   14000208e:	c3                                              	ret
   14000208f:	cc                                              	int3
   140002090:	b0 01                                           	mov    al,0x1
   140002092:	c3                                              	ret
   140002093:	cc                                              	int3
   140002094:	33 c0                                           	xor    eax,eax
   140002096:	c3                                              	ret
   140002097:	cc                                              	int3
   140002098:	cc                                              	int3
   140002099:	cc                                              	int3
   14000209a:	cc                                              	int3
   14000209b:	cc                                              	int3
   14000209c:	cc                                              	int3
   14000209d:	cc                                              	int3
   14000209e:	cc                                              	int3
   14000209f:	cc                                              	int3
   1400020a0:	cc                                              	int3
   1400020a1:	cc                                              	int3
   1400020a2:	cc                                              	int3
   1400020a3:	cc                                              	int3
   1400020a4:	cc                                              	int3
   1400020a5:	cc                                              	int3
   1400020a6:	66 66 0f 1f 84 00 00 00 00 00                   	data16 nop WORD PTR [rax+rax*1+0x0]
   1400020b0:	ff e0                                           	jmp    rax
   1400020b2:	cc                                              	int3
   1400020b3:	cc                                              	int3
   1400020b4:	cc                                              	int3
   1400020b5:	cc                                              	int3
   1400020b6:	cc                                              	int3
   1400020b7:	cc                                              	int3
   1400020b8:	cc                                              	int3
   1400020b9:	cc                                              	int3
   1400020ba:	cc                                              	int3
   1400020bb:	cc                                              	int3
   1400020bc:	cc                                              	int3
   1400020bd:	cc                                              	int3
   1400020be:	cc                                              	int3
   1400020bf:	cc                                              	int3
   1400020c0:	cc                                              	int3
   1400020c1:	cc                                              	int3
   1400020c2:	cc                                              	int3
   1400020c3:	cc                                              	int3
   1400020c4:	cc                                              	int3
   1400020c5:	cc                                              	int3
   1400020c6:	66 66 0f 1f 84 00 00 00 00 00                   	data16 nop WORD PTR [rax+rax*1+0x0]
   1400020d0:	ff 25 3a 12 00 00                               	jmp    QWORD PTR [rip+0x123a]        # 0x140003310
   1400020d6:	40 55                                           	rex push rbp
   1400020d8:	48 83 ec 20                                     	sub    rsp,0x20
   1400020dc:	48 8b ea                                        	mov    rbp,rdx
   1400020df:	48 8b d1                                        	mov    rdx,rcx
   1400020e2:	48 8b 09                                        	mov    rcx,QWORD PTR [rcx]
   1400020e5:	8b 09                                           	mov    ecx,DWORD PTR [rcx]
   1400020e7:	e8 54 02 00 00                                  	call   0x140002340
   1400020ec:	90                                              	nop
   1400020ed:	48 83 c4 20                                     	add    rsp,0x20
   1400020f1:	5d                                              	pop    rbp
   1400020f2:	c3                                              	ret
   1400020f3:	cc                                              	int3
   1400020f4:	40 55                                           	rex push rbp
   1400020f6:	48 8b ea                                        	mov    rbp,rdx
   1400020f9:	48 8b 01                                        	mov    rax,QWORD PTR [rcx]
   1400020fc:	33 c9                                           	xor    ecx,ecx
   1400020fe:	81 38 05 00 00 c0                               	cmp    DWORD PTR [rax],0xc0000005
   140002104:	0f 94 c1                                        	sete   cl
   140002107:	8b c1                                           	mov    eax,ecx
   140002109:	5d                                              	pop    rbp
   14000210a:	c3                                              	ret
   14000210b:	cc                                              	int3
   14000210c:	cc                                              	int3
   14000210d:	cc                                              	int3
   14000210e:	cc                                              	int3
   14000210f:	cc                                              	int3
   140002110:	ff 25 a2 14 00 00                               	jmp    QWORD PTR [rip+0x14a2]        # 0x1400035b8
   140002116:	cc                                              	int3
   140002117:	cc                                              	int3
   140002118:	cc                                              	int3
   140002119:	cc                                              	int3
   14000211a:	cc                                              	int3
   14000211b:	cc                                              	int3
   14000211c:	cc                                              	int3
   14000211d:	cc                                              	int3
   14000211e:	cc                                              	int3
   14000211f:	cc                                              	int3
   140002120:	ff 25 9a 14 00 00                               	jmp    QWORD PTR [rip+0x149a]        # 0x1400035c0
   140002126:	cc                                              	int3
   140002127:	cc                                              	int3
   140002128:	cc                                              	int3
   140002129:	cc                                              	int3
   14000212a:	cc                                              	int3
   14000212b:	cc                                              	int3
   14000212c:	cc                                              	int3
   14000212d:	cc                                              	int3
   14000212e:	cc                                              	int3
   14000212f:	cc                                              	int3
   140002130:	ff 25 92 14 00 00                               	jmp    QWORD PTR [rip+0x1492]        # 0x1400035c8
   140002136:	cc                                              	int3
   140002137:	cc                                              	int3
   140002138:	cc                                              	int3
   140002139:	cc                                              	int3
   14000213a:	cc                                              	int3
   14000213b:	cc                                              	int3
   14000213c:	cc                                              	int3
   14000213d:	cc                                              	int3
   14000213e:	cc                                              	int3
   14000213f:	cc                                              	int3
   140002140:	ff 25 8a 14 00 00                               	jmp    QWORD PTR [rip+0x148a]        # 0x1400035d0
   140002146:	cc                                              	int3
   140002147:	cc                                              	int3
   140002148:	cc                                              	int3
   140002149:	cc                                              	int3
   14000214a:	cc                                              	int3
   14000214b:	cc                                              	int3
   14000214c:	cc                                              	int3
   14000214d:	cc                                              	int3
   14000214e:	cc                                              	int3
   14000214f:	cc                                              	int3
   140002150:	ff 25 82 14 00 00                               	jmp    QWORD PTR [rip+0x1482]        # 0x1400035d8
   140002156:	cc                                              	int3
   140002157:	cc                                              	int3
   140002158:	cc                                              	int3
   140002159:	cc                                              	int3
   14000215a:	cc                                              	int3
   14000215b:	cc                                              	int3
   14000215c:	cc                                              	int3
   14000215d:	cc                                              	int3
   14000215e:	cc                                              	int3
   14000215f:	cc                                              	int3
   140002160:	ff 25 7a 14 00 00                               	jmp    QWORD PTR [rip+0x147a]        # 0x1400035e0
   140002166:	cc                                              	int3
   140002167:	cc                                              	int3
   140002168:	cc                                              	int3
   140002169:	cc                                              	int3
   14000216a:	cc                                              	int3
   14000216b:	cc                                              	int3
   14000216c:	cc                                              	int3
   14000216d:	cc                                              	int3
   14000216e:	cc                                              	int3
   14000216f:	cc                                              	int3
   140002170:	ff 25 72 14 00 00                               	jmp    QWORD PTR [rip+0x1472]        # 0x1400035e8
   140002176:	cc                                              	int3
   140002177:	cc                                              	int3
   140002178:	cc                                              	int3
   140002179:	cc                                              	int3
   14000217a:	cc                                              	int3
   14000217b:	cc                                              	int3
   14000217c:	cc                                              	int3
   14000217d:	cc                                              	int3
   14000217e:	cc                                              	int3
   14000217f:	cc                                              	int3
   140002180:	ff 25 6a 14 00 00                               	jmp    QWORD PTR [rip+0x146a]        # 0x1400035f0
   140002186:	cc                                              	int3
   140002187:	cc                                              	int3
   140002188:	cc                                              	int3
   140002189:	cc                                              	int3
   14000218a:	cc                                              	int3
   14000218b:	cc                                              	int3
   14000218c:	cc                                              	int3
   14000218d:	cc                                              	int3
   14000218e:	cc                                              	int3
   14000218f:	cc                                              	int3
   140002190:	ff 25 6a 14 00 00                               	jmp    QWORD PTR [rip+0x146a]        # 0x140003600
   140002196:	cc                                              	int3
   140002197:	cc                                              	int3
   140002198:	cc                                              	int3
   140002199:	cc                                              	int3
   14000219a:	cc                                              	int3
   14000219b:	cc                                              	int3
   14000219c:	cc                                              	int3
   14000219d:	cc                                              	int3
   14000219e:	cc                                              	int3
   14000219f:	cc                                              	int3
   1400021a0:	ff 25 62 14 00 00                               	jmp    QWORD PTR [rip+0x1462]        # 0x140003608
   1400021a6:	cc                                              	int3
   1400021a7:	cc                                              	int3
   1400021a8:	cc                                              	int3
   1400021a9:	cc                                              	int3
   1400021aa:	cc                                              	int3
   1400021ab:	cc                                              	int3
   1400021ac:	cc                                              	int3
   1400021ad:	cc                                              	int3
   1400021ae:	cc                                              	int3
   1400021af:	cc                                              	int3
   1400021b0:	ff 25 5a 14 00 00                               	jmp    QWORD PTR [rip+0x145a]        # 0x140003610
   1400021b6:	cc                                              	int3
   1400021b7:	cc                                              	int3
   1400021b8:	cc                                              	int3
   1400021b9:	cc                                              	int3
   1400021ba:	cc                                              	int3
   1400021bb:	cc                                              	int3
   1400021bc:	cc                                              	int3
   1400021bd:	cc                                              	int3
   1400021be:	cc                                              	int3
   1400021bf:	cc                                              	int3
   1400021c0:	ff 25 52 14 00 00                               	jmp    QWORD PTR [rip+0x1452]        # 0x140003618
   1400021c6:	cc                                              	int3
   1400021c7:	cc                                              	int3
   1400021c8:	cc                                              	int3
   1400021c9:	cc                                              	int3
   1400021ca:	cc                                              	int3
   1400021cb:	cc                                              	int3
   1400021cc:	cc                                              	int3
   1400021cd:	cc                                              	int3
   1400021ce:	cc                                              	int3
   1400021cf:	cc                                              	int3
   1400021d0:	ff 25 52 14 00 00                               	jmp    QWORD PTR [rip+0x1452]        # 0x140003628
   1400021d6:	cc                                              	int3
   1400021d7:	cc                                              	int3
   1400021d8:	cc                                              	int3
   1400021d9:	cc                                              	int3
   1400021da:	cc                                              	int3
   1400021db:	cc                                              	int3
   1400021dc:	cc                                              	int3
   1400021dd:	cc                                              	int3
   1400021de:	cc                                              	int3
   1400021df:	cc                                              	int3
   1400021e0:	ff 25 52 14 00 00                               	jmp    QWORD PTR [rip+0x1452]        # 0x140003638
   1400021e6:	cc                                              	int3
   1400021e7:	cc                                              	int3
   1400021e8:	cc                                              	int3
   1400021e9:	cc                                              	int3
   1400021ea:	cc                                              	int3
   1400021eb:	cc                                              	int3
   1400021ec:	cc                                              	int3
   1400021ed:	cc                                              	int3
   1400021ee:	cc                                              	int3
   1400021ef:	cc                                              	int3
   1400021f0:	ff 25 52 14 00 00                               	jmp    QWORD PTR [rip+0x1452]        # 0x140003648
   1400021f6:	cc                                              	int3
   1400021f7:	cc                                              	int3
   1400021f8:	cc                                              	int3
   1400021f9:	cc                                              	int3
   1400021fa:	cc                                              	int3
   1400021fb:	cc                                              	int3
   1400021fc:	cc                                              	int3
   1400021fd:	cc                                              	int3
   1400021fe:	cc                                              	int3
   1400021ff:	cc                                              	int3
   140002200:	ff 25 5a 14 00 00                               	jmp    QWORD PTR [rip+0x145a]        # 0x140003660
   140002206:	cc                                              	int3
   140002207:	cc                                              	int3
   140002208:	cc                                              	int3
   140002209:	cc                                              	int3
   14000220a:	cc                                              	int3
   14000220b:	cc                                              	int3
   14000220c:	cc                                              	int3
   14000220d:	cc                                              	int3
   14000220e:	cc                                              	int3
   14000220f:	cc                                              	int3
   140002210:	ff 25 52 14 00 00                               	jmp    QWORD PTR [rip+0x1452]        # 0x140003668
   140002216:	cc                                              	int3
   140002217:	cc                                              	int3
   140002218:	cc                                              	int3
   140002219:	cc                                              	int3
   14000221a:	cc                                              	int3
   14000221b:	cc                                              	int3
   14000221c:	cc                                              	int3
   14000221d:	cc                                              	int3
   14000221e:	cc                                              	int3
   14000221f:	cc                                              	int3
   140002220:	ff 25 1a 14 00 00                               	jmp    QWORD PTR [rip+0x141a]        # 0x140003640
   140002226:	cc                                              	int3
   140002227:	cc                                              	int3
   140002228:	cc                                              	int3
   140002229:	cc                                              	int3
   14000222a:	cc                                              	int3
   14000222b:	cc                                              	int3
   14000222c:	cc                                              	int3
   14000222d:	cc                                              	int3
   14000222e:	cc                                              	int3
   14000222f:	cc                                              	int3
   140002230:	ff 25 da 14 00 00                               	jmp    QWORD PTR [rip+0x14da]        # 0x140003710
   140002236:	cc                                              	int3
   140002237:	cc                                              	int3
   140002238:	cc                                              	int3
   140002239:	cc                                              	int3
   14000223a:	cc                                              	int3
   14000223b:	cc                                              	int3
   14000223c:	cc                                              	int3
   14000223d:	cc                                              	int3
   14000223e:	cc                                              	int3
   14000223f:	cc                                              	int3
   140002240:	ff 25 2a 14 00 00                               	jmp    QWORD PTR [rip+0x142a]        # 0x140003670
   140002246:	cc                                              	int3
   140002247:	cc                                              	int3
   140002248:	cc                                              	int3
   140002249:	cc                                              	int3
   14000224a:	cc                                              	int3
   14000224b:	cc                                              	int3
   14000224c:	cc                                              	int3
   14000224d:	cc                                              	int3
   14000224e:	cc                                              	int3
   14000224f:	cc                                              	int3
   140002250:	ff 25 22 14 00 00                               	jmp    QWORD PTR [rip+0x1422]        # 0x140003678
   140002256:	cc                                              	int3
   140002257:	cc                                              	int3
   140002258:	cc                                              	int3
   140002259:	cc                                              	int3
   14000225a:	cc                                              	int3
   14000225b:	cc                                              	int3
   14000225c:	cc                                              	int3
   14000225d:	cc                                              	int3
   14000225e:	cc                                              	int3
   14000225f:	cc                                              	int3
   140002260:	ff 25 ba 14 00 00                               	jmp    QWORD PTR [rip+0x14ba]        # 0x140003720
   140002266:	cc                                              	int3
   140002267:	cc                                              	int3
   140002268:	cc                                              	int3
   140002269:	cc                                              	int3
   14000226a:	cc                                              	int3
   14000226b:	cc                                              	int3
   14000226c:	cc                                              	int3
   14000226d:	cc                                              	int3
   14000226e:	cc                                              	int3
   14000226f:	cc                                              	int3
   140002270:	ff 25 0a 14 00 00                               	jmp    QWORD PTR [rip+0x140a]        # 0x140003680
   140002276:	cc                                              	int3
   140002277:	cc                                              	int3
   140002278:	cc                                              	int3
   140002279:	cc                                              	int3
   14000227a:	cc                                              	int3
   14000227b:	cc                                              	int3
   14000227c:	cc                                              	int3
   14000227d:	cc                                              	int3
   14000227e:	cc                                              	int3
   14000227f:	cc                                              	int3
   140002280:	ff 25 02 14 00 00                               	jmp    QWORD PTR [rip+0x1402]        # 0x140003688
   140002286:	cc                                              	int3
   140002287:	cc                                              	int3
   140002288:	cc                                              	int3
   140002289:	cc                                              	int3
   14000228a:	cc                                              	int3
   14000228b:	cc                                              	int3
   14000228c:	cc                                              	int3
   14000228d:	cc                                              	int3
   14000228e:	cc                                              	int3
   14000228f:	cc                                              	int3
   140002290:	ff 25 fa 13 00 00                               	jmp    QWORD PTR [rip+0x13fa]        # 0x140003690
   140002296:	cc                                              	int3
   140002297:	cc                                              	int3
   140002298:	cc                                              	int3
   140002299:	cc                                              	int3
   14000229a:	cc                                              	int3
   14000229b:	cc                                              	int3
   14000229c:	cc                                              	int3
   14000229d:	cc                                              	int3
   14000229e:	cc                                              	int3
   14000229f:	cc                                              	int3
   1400022a0:	ff 25 f2 13 00 00                               	jmp    QWORD PTR [rip+0x13f2]        # 0x140003698
   1400022a6:	cc                                              	int3
   1400022a7:	cc                                              	int3
   1400022a8:	cc                                              	int3
   1400022a9:	cc                                              	int3
   1400022aa:	cc                                              	int3
   1400022ab:	cc                                              	int3
   1400022ac:	cc                                              	int3
   1400022ad:	cc                                              	int3
   1400022ae:	cc                                              	int3
   1400022af:	cc                                              	int3
   1400022b0:	ff 25 ea 13 00 00                               	jmp    QWORD PTR [rip+0x13ea]        # 0x1400036a0
   1400022b6:	cc                                              	int3
   1400022b7:	cc                                              	int3
   1400022b8:	cc                                              	int3
   1400022b9:	cc                                              	int3
   1400022ba:	cc                                              	int3
   1400022bb:	cc                                              	int3
   1400022bc:	cc                                              	int3
   1400022bd:	cc                                              	int3
   1400022be:	cc                                              	int3
   1400022bf:	cc                                              	int3
   1400022c0:	ff 25 e2 13 00 00                               	jmp    QWORD PTR [rip+0x13e2]        # 0x1400036a8
   1400022c6:	cc                                              	int3
   1400022c7:	cc                                              	int3
   1400022c8:	cc                                              	int3
   1400022c9:	cc                                              	int3
   1400022ca:	cc                                              	int3
   1400022cb:	cc                                              	int3
   1400022cc:	cc                                              	int3
   1400022cd:	cc                                              	int3
   1400022ce:	cc                                              	int3
   1400022cf:	cc                                              	int3
   1400022d0:	ff 25 da 13 00 00                               	jmp    QWORD PTR [rip+0x13da]        # 0x1400036b0
   1400022d6:	cc                                              	int3
   1400022d7:	cc                                              	int3
   1400022d8:	cc                                              	int3
   1400022d9:	cc                                              	int3
   1400022da:	cc                                              	int3
   1400022db:	cc                                              	int3
   1400022dc:	cc                                              	int3
   1400022dd:	cc                                              	int3
   1400022de:	cc                                              	int3
   1400022df:	cc                                              	int3
   1400022e0:	ff 25 d2 13 00 00                               	jmp    QWORD PTR [rip+0x13d2]        # 0x1400036b8
   1400022e6:	cc                                              	int3
   1400022e7:	cc                                              	int3
   1400022e8:	cc                                              	int3
   1400022e9:	cc                                              	int3
   1400022ea:	cc                                              	int3
   1400022eb:	cc                                              	int3
   1400022ec:	cc                                              	int3
   1400022ed:	cc                                              	int3
   1400022ee:	cc                                              	int3
   1400022ef:	cc                                              	int3
   1400022f0:	ff 25 ca 13 00 00                               	jmp    QWORD PTR [rip+0x13ca]        # 0x1400036c0
   1400022f6:	cc                                              	int3
   1400022f7:	cc                                              	int3
   1400022f8:	cc                                              	int3
   1400022f9:	cc                                              	int3
   1400022fa:	cc                                              	int3
   1400022fb:	cc                                              	int3
   1400022fc:	cc                                              	int3
   1400022fd:	cc                                              	int3
   1400022fe:	cc                                              	int3
   1400022ff:	cc                                              	int3
   140002300:	ff 25 c2 13 00 00                               	jmp    QWORD PTR [rip+0x13c2]        # 0x1400036c8
   140002306:	cc                                              	int3
   140002307:	cc                                              	int3
   140002308:	cc                                              	int3
   140002309:	cc                                              	int3
   14000230a:	cc                                              	int3
   14000230b:	cc                                              	int3
   14000230c:	cc                                              	int3
   14000230d:	cc                                              	int3
   14000230e:	cc                                              	int3
   14000230f:	cc                                              	int3
   140002310:	ff 25 ba 13 00 00                               	jmp    QWORD PTR [rip+0x13ba]        # 0x1400036d0
   140002316:	cc                                              	int3
   140002317:	cc                                              	int3
   140002318:	cc                                              	int3
   140002319:	cc                                              	int3
   14000231a:	cc                                              	int3
   14000231b:	cc                                              	int3
   14000231c:	cc                                              	int3
   14000231d:	cc                                              	int3
   14000231e:	cc                                              	int3
   14000231f:	cc                                              	int3
   140002320:	ff 25 b2 13 00 00                               	jmp    QWORD PTR [rip+0x13b2]        # 0x1400036d8
   140002326:	cc                                              	int3
   140002327:	cc                                              	int3
   140002328:	cc                                              	int3
   140002329:	cc                                              	int3
   14000232a:	cc                                              	int3
   14000232b:	cc                                              	int3
   14000232c:	cc                                              	int3
   14000232d:	cc                                              	int3
   14000232e:	cc                                              	int3
   14000232f:	cc                                              	int3
   140002330:	ff 25 aa 13 00 00                               	jmp    QWORD PTR [rip+0x13aa]        # 0x1400036e0
   140002336:	cc                                              	int3
   140002337:	cc                                              	int3
   140002338:	cc                                              	int3
   140002339:	cc                                              	int3
   14000233a:	cc                                              	int3
   14000233b:	cc                                              	int3
   14000233c:	cc                                              	int3
   14000233d:	cc                                              	int3
   14000233e:	cc                                              	int3
   14000233f:	cc                                              	int3
   140002340:	ff 25 a2 13 00 00                               	jmp    QWORD PTR [rip+0x13a2]        # 0x1400036e8
   140002346:	cc                                              	int3
   140002347:	cc                                              	int3
   140002348:	cc                                              	int3
   140002349:	cc                                              	int3
   14000234a:	cc                                              	int3
   14000234b:	cc                                              	int3
   14000234c:	cc                                              	int3
   14000234d:	cc                                              	int3
   14000234e:	cc                                              	int3
   14000234f:	cc                                              	int3
   140002350:	ff 25 9a 13 00 00                               	jmp    QWORD PTR [rip+0x139a]        # 0x1400036f0
   140002356:	cc                                              	int3
   140002357:	cc                                              	int3
   140002358:	cc                                              	int3
   140002359:	cc                                              	int3
   14000235a:	cc                                              	int3
   14000235b:	cc                                              	int3
   14000235c:	cc                                              	int3
   14000235d:	cc                                              	int3
   14000235e:	cc                                              	int3
   14000235f:	cc                                              	int3
   140002360:	ff 25 ea 12 00 00                               	jmp    QWORD PTR [rip+0x12ea]        # 0x140003650
   140002366:	cc                                              	int3
   140002367:	cc                                              	int3
   140002368:	cc                                              	int3
   140002369:	cc                                              	int3
   14000236a:	cc                                              	int3
   14000236b:	cc                                              	int3
   14000236c:	cc                                              	int3
   14000236d:	cc                                              	int3
   14000236e:	cc                                              	int3
   14000236f:	cc                                              	int3
   140002370:	ff 25 ba 13 00 00                               	jmp    QWORD PTR [rip+0x13ba]        # 0x140003730
   140002376:	cc                                              	int3
   140002377:	cc                                              	int3
   140002378:	cc                                              	int3
   140002379:	cc                                              	int3
   14000237a:	cc                                              	int3
   14000237b:	cc                                              	int3
   14000237c:	cc                                              	int3
   14000237d:	cc                                              	int3
   14000237e:	cc                                              	int3
   14000237f:	cc                                              	int3
   140002380:	ff 25 72 13 00 00                               	jmp    QWORD PTR [rip+0x1372]        # 0x1400036f8
   140002386:	cc                                              	int3
   140002387:	cc                                              	int3
   140002388:	cc                                              	int3
   140002389:	cc                                              	int3
   14000238a:	cc                                              	int3
   14000238b:	cc                                              	int3
   14000238c:	cc                                              	int3
   14000238d:	cc                                              	int3
   14000238e:	cc                                              	int3
   14000238f:	cc                                              	int3
   140002390:	ff 25 6a 13 00 00                               	jmp    QWORD PTR [rip+0x136a]        # 0x140003700
   140002396:	cc                                              	int3
   140002397:	cc                                              	int3
   140002398:	cc                                              	int3
   140002399:	cc                                              	int3
   14000239a:	cc                                              	int3
   14000239b:	cc                                              	int3
   14000239c:	cc                                              	int3
   14000239d:	cc                                              	int3
   14000239e:	cc                                              	int3
   14000239f:	cc                                              	int3
   1400023a0:	ff 25 7a 12 00 00                               	jmp    QWORD PTR [rip+0x127a]        # 0x140003620

