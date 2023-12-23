
output.s:     file format pei-x86-64


Disassembly of section .text:

0000000140001000 <__mingw_invalidParameterHandler>:
   140001000:	55                                              	push   rbp
   140001001:	48 89 e5                                        	mov    rbp,rsp
   140001004:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140001008:	48 89 55 18                                     	mov    QWORD PTR [rbp+0x18],rdx
   14000100c:	4c 89 45 20                                     	mov    QWORD PTR [rbp+0x20],r8
   140001010:	44 89 4d 28                                     	mov    DWORD PTR [rbp+0x28],r9d
   140001014:	90                                              	nop
   140001015:	5d                                              	pop    rbp
   140001016:	c3                                              	ret

0000000140001017 <pre_c_init>:
   140001017:	55                                              	push   rbp
   140001018:	48 89 e5                                        	mov    rbp,rsp
   14000101b:	48 83 ec 20                                     	sub    rsp,0x20
   14000101f:	e8 5b 04 00 00                                  	call   14000147f <check_managed_app>
   140001024:	89 05 f6 7f 00 00                               	mov    DWORD PTR [rip+0x7ff6],eax        # 140009020 <managedapp>
   14000102a:	48 8b 05 ff 43 00 00                            	mov    rax,QWORD PTR [rip+0x43ff]        # 140005430 <.refptr.__mingw_app_type>
   140001031:	8b 00                                           	mov    eax,DWORD PTR [rax]
   140001033:	85 c0                                           	test   eax,eax
   140001035:	74 0c                                           	je     140001043 <pre_c_init+0x2c>
   140001037:	b9 02 00 00 00                                  	mov    ecx,0x2
   14000103c:	e8 1f 22 00 00                                  	call   140003260 <__set_app_type>
   140001041:	eb 0a                                           	jmp    14000104d <pre_c_init+0x36>
   140001043:	b9 01 00 00 00                                  	mov    ecx,0x1
   140001048:	e8 13 22 00 00                                  	call   140003260 <__set_app_type>
   14000104d:	e8 96 21 00 00                                  	call   1400031e8 <__p__fmode>
   140001052:	48 8b 15 b7 44 00 00                            	mov    rdx,QWORD PTR [rip+0x44b7]        # 140005510 <.refptr._fmode>
   140001059:	8b 12                                           	mov    edx,DWORD PTR [rdx]
   14000105b:	89 10                                           	mov    DWORD PTR [rax],edx
   14000105d:	e8 76 21 00 00                                  	call   1400031d8 <__p__commode>
   140001062:	48 8b 15 87 44 00 00                            	mov    rdx,QWORD PTR [rip+0x4487]        # 1400054f0 <.refptr._commode>
   140001069:	8b 12                                           	mov    edx,DWORD PTR [rdx]
   14000106b:	89 10                                           	mov    DWORD PTR [rax],edx
   14000106d:	e8 7e 08 00 00                                  	call   1400018f0 <_setargv>
   140001072:	48 8b 05 27 43 00 00                            	mov    rax,QWORD PTR [rip+0x4327]        # 1400053a0 <.refptr._MINGW_INSTALL_DEBUG_MATHERR>
   140001079:	8b 00                                           	mov    eax,DWORD PTR [rax]
   14000107b:	83 f8 01                                        	cmp    eax,0x1
   14000107e:	75 0f                                           	jne    14000108f <pre_c_init+0x78>
   140001080:	48 8b 05 a9 44 00 00                            	mov    rax,QWORD PTR [rip+0x44a9]        # 140005530 <.refptr._matherr>
   140001087:	48 89 c1                                        	mov    rcx,rax
   14000108a:	e8 bb 13 00 00                                  	call   14000244a <__mingw_setusermatherr>
   14000108f:	b8 00 00 00 00                                  	mov    eax,0x0
   140001094:	48 83 c4 20                                     	add    rsp,0x20
   140001098:	5d                                              	pop    rbp
   140001099:	c3                                              	ret

000000014000109a <pre_cpp_init>:
   14000109a:	55                                              	push   rbp
   14000109b:	48 89 e5                                        	mov    rbp,rsp
   14000109e:	48 83 ec 30                                     	sub    rsp,0x30
   1400010a2:	48 8b 05 97 44 00 00                            	mov    rax,QWORD PTR [rip+0x4497]        # 140005540 <.refptr._newmode>
   1400010a9:	8b 00                                           	mov    eax,DWORD PTR [rax]
   1400010ab:	89 05 77 7f 00 00                               	mov    DWORD PTR [rip+0x7f77],eax        # 140009028 <startinfo>
   1400010b1:	48 8b 05 48 44 00 00                            	mov    rax,QWORD PTR [rip+0x4448]        # 140005500 <.refptr._dowildcard>
   1400010b8:	8b 10                                           	mov    edx,DWORD PTR [rax]
   1400010ba:	48 8d 05 67 7f 00 00                            	lea    rax,[rip+0x7f67]        # 140009028 <startinfo>
   1400010c1:	48 89 44 24 20                                  	mov    QWORD PTR [rsp+0x20],rax
   1400010c6:	41 89 d1                                        	mov    r9d,edx
   1400010c9:	4c 8d 05 40 7f 00 00                            	lea    r8,[rip+0x7f40]        # 140009010 <envp>
   1400010d0:	48 8d 05 31 7f 00 00                            	lea    rax,[rip+0x7f31]        # 140009008 <argv>
   1400010d7:	48 89 c2                                        	mov    rdx,rax
   1400010da:	48 8d 05 23 7f 00 00                            	lea    rax,[rip+0x7f23]        # 140009004 <argc>
   1400010e1:	48 89 c1                                        	mov    rcx,rax
   1400010e4:	e8 67 1e 00 00                                  	call   140002f50 <__getmainargs>
   1400010e9:	89 05 29 7f 00 00                               	mov    DWORD PTR [rip+0x7f29],eax        # 140009018 <argret>
   1400010ef:	90                                              	nop
   1400010f0:	48 83 c4 30                                     	add    rsp,0x30
   1400010f4:	5d                                              	pop    rbp
   1400010f5:	c3                                              	ret

00000001400010f6 <WinMainCRTStartup>:
   1400010f6:	55                                              	push   rbp
   1400010f7:	48 89 e5                                        	mov    rbp,rsp
   1400010fa:	48 83 ec 30                                     	sub    rsp,0x30
   1400010fe:	c7 45 fc ff 00 00 00                            	mov    DWORD PTR [rbp-0x4],0xff

0000000140001105 <.l_startw>:
   140001105:	48 8b 05 24 43 00 00                            	mov    rax,QWORD PTR [rip+0x4324]        # 140005430 <.refptr.__mingw_app_type>
   14000110c:	c7 00 01 00 00 00                               	mov    DWORD PTR [rax],0x1
   140001112:	e8 3d 00 00 00                                  	call   140001154 <__tmainCRTStartup>
   140001117:	89 45 fc                                        	mov    DWORD PTR [rbp-0x4],eax
   14000111a:	90                                              	nop

000000014000111b <.l_endw>:
   14000111b:	90                                              	nop
   14000111c:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   14000111f:	48 83 c4 30                                     	add    rsp,0x30
   140001123:	5d                                              	pop    rbp
   140001124:	c3                                              	ret

0000000140001125 <mainCRTStartup>:
   140001125:	55                                              	push   rbp
   140001126:	48 89 e5                                        	mov    rbp,rsp
   140001129:	48 83 ec 30                                     	sub    rsp,0x30
   14000112d:	c7 45 fc ff 00 00 00                            	mov    DWORD PTR [rbp-0x4],0xff

0000000140001134 <.l_start>:
   140001134:	48 8b 05 f5 42 00 00                            	mov    rax,QWORD PTR [rip+0x42f5]        # 140005430 <.refptr.__mingw_app_type>
   14000113b:	c7 00 00 00 00 00                               	mov    DWORD PTR [rax],0x0
   140001141:	e8 0e 00 00 00                                  	call   140001154 <__tmainCRTStartup>
   140001146:	89 45 fc                                        	mov    DWORD PTR [rbp-0x4],eax
   140001149:	90                                              	nop

000000014000114a <.l_end>:
   14000114a:	90                                              	nop
   14000114b:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   14000114e:	48 83 c4 30                                     	add    rsp,0x30
   140001152:	5d                                              	pop    rbp
   140001153:	c3                                              	ret

0000000140001154 <__tmainCRTStartup>:
   140001154:	55                                              	push   rbp
   140001155:	48 89 e5                                        	mov    rbp,rsp
   140001158:	48 81 ec e0 00 00 00                            	sub    rsp,0xe0
   14000115f:	48 c7 45 f8 00 00 00 00                         	mov    QWORD PTR [rbp-0x8],0x0
   140001167:	c7 45 f4 00 00 00 00                            	mov    DWORD PTR [rbp-0xc],0x0
   14000116e:	48 8d 85 40 ff ff ff                            	lea    rax,[rbp-0xc0]
   140001175:	41 b8 68 00 00 00                               	mov    r8d,0x68
   14000117b:	ba 00 00 00 00                                  	mov    edx,0x0
   140001180:	48 89 c1                                        	mov    rcx,rax
   140001183:	e8 28 21 00 00                                  	call   1400032b0 <memset>
   140001188:	48 8b 05 a1 42 00 00                            	mov    rax,QWORD PTR [rip+0x42a1]        # 140005430 <.refptr.__mingw_app_type>
   14000118f:	8b 00                                           	mov    eax,DWORD PTR [rax]
   140001191:	85 c0                                           	test   eax,eax
   140001193:	74 13                                           	je     1400011a8 <__tmainCRTStartup+0x54>
   140001195:	48 8d 85 40 ff ff ff                            	lea    rax,[rbp-0xc0]
   14000119c:	48 89 c1                                        	mov    rcx,rax
   14000119f:	48 8b 05 8a 90 00 00                            	mov    rax,QWORD PTR [rip+0x908a]        # 14000a230 <__imp_GetStartupInfoA>
   1400011a6:	ff d0                                           	call   rax
   1400011a8:	48 c7 45 e8 00 00 00 00                         	mov    QWORD PTR [rbp-0x18],0x0
   1400011b0:	c7 45 dc 30 00 00 00                            	mov    DWORD PTR [rbp-0x24],0x30
   1400011b7:	8b 45 dc                                        	mov    eax,DWORD PTR [rbp-0x24]
   1400011ba:	65 48 8b 00                                     	mov    rax,QWORD PTR gs:[rax]
   1400011be:	48 89 45 d0                                     	mov    QWORD PTR [rbp-0x30],rax
   1400011c2:	48 8b 45 d0                                     	mov    rax,QWORD PTR [rbp-0x30]
   1400011c6:	48 8b 40 08                                     	mov    rax,QWORD PTR [rax+0x8]
   1400011ca:	48 89 45 e0                                     	mov    QWORD PTR [rbp-0x20],rax
   1400011ce:	c7 45 f0 00 00 00 00                            	mov    DWORD PTR [rbp-0x10],0x0
   1400011d5:	eb 21                                           	jmp    1400011f8 <__tmainCRTStartup+0xa4>
   1400011d7:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   1400011db:	48 3b 45 e0                                     	cmp    rax,QWORD PTR [rbp-0x20]
   1400011df:	75 09                                           	jne    1400011ea <__tmainCRTStartup+0x96>
   1400011e1:	c7 45 f0 01 00 00 00                            	mov    DWORD PTR [rbp-0x10],0x1
   1400011e8:	eb 45                                           	jmp    14000122f <__tmainCRTStartup+0xdb>
   1400011ea:	b9 e8 03 00 00                                  	mov    ecx,0x3e8
   1400011ef:	48 8b 05 62 90 00 00                            	mov    rax,QWORD PTR [rip+0x9062]        # 14000a258 <__imp_Sleep>
   1400011f6:	ff d0                                           	call   rax
   1400011f8:	48 8b 05 91 42 00 00                            	mov    rax,QWORD PTR [rip+0x4291]        # 140005490 <.refptr.__native_startup_lock>
   1400011ff:	48 89 45 c8                                     	mov    QWORD PTR [rbp-0x38],rax
   140001203:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140001207:	48 89 45 c0                                     	mov    QWORD PTR [rbp-0x40],rax
   14000120b:	48 c7 45 b8 00 00 00 00                         	mov    QWORD PTR [rbp-0x48],0x0
   140001213:	48 8b 4d c0                                     	mov    rcx,QWORD PTR [rbp-0x40]
   140001217:	48 8b 45 b8                                     	mov    rax,QWORD PTR [rbp-0x48]
   14000121b:	48 8b 55 c8                                     	mov    rdx,QWORD PTR [rbp-0x38]
   14000121f:	f0 48 0f b1 0a                                  	lock cmpxchg QWORD PTR [rdx],rcx
   140001224:	48 89 45 e8                                     	mov    QWORD PTR [rbp-0x18],rax
   140001228:	48 83 7d e8 00                                  	cmp    QWORD PTR [rbp-0x18],0x0
   14000122d:	75 a8                                           	jne    1400011d7 <__tmainCRTStartup+0x83>
   14000122f:	48 8b 05 6a 42 00 00                            	mov    rax,QWORD PTR [rip+0x426a]        # 1400054a0 <.refptr.__native_startup_state>
   140001236:	8b 00                                           	mov    eax,DWORD PTR [rax]
   140001238:	83 f8 01                                        	cmp    eax,0x1
   14000123b:	75 0c                                           	jne    140001249 <__tmainCRTStartup+0xf5>
   14000123d:	b9 1f 00 00 00                                  	mov    ecx,0x1f
   140001242:	e8 69 1e 00 00                                  	call   1400030b0 <_amsg_exit>
   140001247:	eb 3f                                           	jmp    140001288 <__tmainCRTStartup+0x134>
   140001249:	48 8b 05 50 42 00 00                            	mov    rax,QWORD PTR [rip+0x4250]        # 1400054a0 <.refptr.__native_startup_state>
   140001250:	8b 00                                           	mov    eax,DWORD PTR [rax]
   140001252:	85 c0                                           	test   eax,eax
   140001254:	75 28                                           	jne    14000127e <__tmainCRTStartup+0x12a>
   140001256:	48 8b 05 43 42 00 00                            	mov    rax,QWORD PTR [rip+0x4243]        # 1400054a0 <.refptr.__native_startup_state>
   14000125d:	c7 00 01 00 00 00                               	mov    DWORD PTR [rax],0x1
   140001263:	48 8b 05 76 42 00 00                            	mov    rax,QWORD PTR [rip+0x4276]        # 1400054e0 <.refptr.__xi_z>
   14000126a:	48 89 c2                                        	mov    rdx,rax
   14000126d:	48 8b 05 5c 42 00 00                            	mov    rax,QWORD PTR [rip+0x425c]        # 1400054d0 <.refptr.__xi_a>
   140001274:	48 89 c1                                        	mov    rcx,rax
   140001277:	e8 dc 1f 00 00                                  	call   140003258 <_initterm>
   14000127c:	eb 0a                                           	jmp    140001288 <__tmainCRTStartup+0x134>
   14000127e:	c7 05 9c 7d 00 00 01 00 00 00                   	mov    DWORD PTR [rip+0x7d9c],0x1        # 140009024 <has_cctor>
   140001288:	48 8b 05 11 42 00 00                            	mov    rax,QWORD PTR [rip+0x4211]        # 1400054a0 <.refptr.__native_startup_state>
   14000128f:	8b 00                                           	mov    eax,DWORD PTR [rax]
   140001291:	83 f8 01                                        	cmp    eax,0x1
   140001294:	75 26                                           	jne    1400012bc <__tmainCRTStartup+0x168>
   140001296:	48 8b 05 23 42 00 00                            	mov    rax,QWORD PTR [rip+0x4223]        # 1400054c0 <.refptr.__xc_z>
   14000129d:	48 89 c2                                        	mov    rdx,rax
   1400012a0:	48 8b 05 09 42 00 00                            	mov    rax,QWORD PTR [rip+0x4209]        # 1400054b0 <.refptr.__xc_a>
   1400012a7:	48 89 c1                                        	mov    rcx,rax
   1400012aa:	e8 a9 1f 00 00                                  	call   140003258 <_initterm>
   1400012af:	48 8b 05 ea 41 00 00                            	mov    rax,QWORD PTR [rip+0x41ea]        # 1400054a0 <.refptr.__native_startup_state>
   1400012b6:	c7 00 02 00 00 00                               	mov    DWORD PTR [rax],0x2
   1400012bc:	83 7d f0 00                                     	cmp    DWORD PTR [rbp-0x10],0x0
   1400012c0:	75 1e                                           	jne    1400012e0 <__tmainCRTStartup+0x18c>
   1400012c2:	48 8b 05 c7 41 00 00                            	mov    rax,QWORD PTR [rip+0x41c7]        # 140005490 <.refptr.__native_startup_lock>
   1400012c9:	48 89 45 b0                                     	mov    QWORD PTR [rbp-0x50],rax
   1400012cd:	48 c7 45 a8 00 00 00 00                         	mov    QWORD PTR [rbp-0x58],0x0
   1400012d5:	48 8b 55 a8                                     	mov    rdx,QWORD PTR [rbp-0x58]
   1400012d9:	48 8b 45 b0                                     	mov    rax,QWORD PTR [rbp-0x50]
   1400012dd:	48 87 10                                        	xchg   QWORD PTR [rax],rdx
   1400012e0:	48 8b 05 f9 40 00 00                            	mov    rax,QWORD PTR [rip+0x40f9]        # 1400053e0 <.refptr.__dyn_tls_init_callback>
   1400012e7:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   1400012ea:	48 85 c0                                        	test   rax,rax
   1400012ed:	74 1c                                           	je     14000130b <__tmainCRTStartup+0x1b7>
   1400012ef:	48 8b 05 ea 40 00 00                            	mov    rax,QWORD PTR [rip+0x40ea]        # 1400053e0 <.refptr.__dyn_tls_init_callback>
   1400012f6:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   1400012f9:	41 b8 00 00 00 00                               	mov    r8d,0x0
   1400012ff:	ba 02 00 00 00                                  	mov    edx,0x2
   140001304:	b9 00 00 00 00                                  	mov    ecx,0x0
   140001309:	ff d0                                           	call   rax
   14000130b:	e8 28 10 00 00                                  	call   140002338 <_pei386_runtime_relocator>
   140001310:	48 8b 05 09 42 00 00                            	mov    rax,QWORD PTR [rip+0x4209]        # 140005520 <.refptr._gnu_exception_handler>
   140001317:	48 89 c1                                        	mov    rcx,rax
   14000131a:	48 8b 05 2f 8f 00 00                            	mov    rax,QWORD PTR [rip+0x8f2f]        # 14000a250 <__imp_SetUnhandledExceptionFilter>
   140001321:	ff d0                                           	call   rax
   140001323:	48 8b 15 56 41 00 00                            	mov    rdx,QWORD PTR [rip+0x4156]        # 140005480 <.refptr.__mingw_oldexcpt_handler>
   14000132a:	48 89 02                                        	mov    QWORD PTR [rdx],rax
   14000132d:	48 8d 05 cc fc ff ff                            	lea    rax,[rip+0xfffffffffffffccc]        # 140001000 <__mingw_invalidParameterHandler>
   140001334:	48 89 c1                                        	mov    rcx,rax
   140001337:	e8 2c 1f 00 00                                  	call   140003268 <_set_invalid_parameter_handler>
   14000133c:	e8 ef 07 00 00                                  	call   140001b30 <_fpreset>
   140001341:	48 8d 05 80 7e 00 00                            	lea    rax,[rip+0x7e80]        # 1400091c8 <__mingw_winmain_hInstance>
   140001348:	48 8b 15 b1 40 00 00                            	mov    rdx,QWORD PTR [rip+0x40b1]        # 140005400 <.refptr.__image_base__>
   14000134f:	48 89 10                                        	mov    QWORD PTR [rax],rdx
   140001352:	e8 79 1e 00 00                                  	call   1400031d0 <__p__acmdln>
   140001357:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   14000135a:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   14000135e:	48 83 7d f8 00                                  	cmp    QWORD PTR [rbp-0x8],0x0
   140001363:	74 66                                           	je     1400013cb <__tmainCRTStartup+0x277>
   140001365:	eb 1d                                           	jmp    140001384 <__tmainCRTStartup+0x230>
   140001367:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   14000136b:	0f b6 00                                        	movzx  eax,BYTE PTR [rax]
   14000136e:	3c 22                                           	cmp    al,0x22
   140001370:	75 0d                                           	jne    14000137f <__tmainCRTStartup+0x22b>
   140001372:	83 7d f4 00                                     	cmp    DWORD PTR [rbp-0xc],0x0
   140001376:	0f 94 c0                                        	sete   al
   140001379:	0f b6 c0                                        	movzx  eax,al
   14000137c:	89 45 f4                                        	mov    DWORD PTR [rbp-0xc],eax
   14000137f:	48 83 45 f8 01                                  	add    QWORD PTR [rbp-0x8],0x1
   140001384:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140001388:	0f b6 00                                        	movzx  eax,BYTE PTR [rax]
   14000138b:	3c 20                                           	cmp    al,0x20
   14000138d:	7f d8                                           	jg     140001367 <__tmainCRTStartup+0x213>
   14000138f:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140001393:	0f b6 00                                        	movzx  eax,BYTE PTR [rax]
   140001396:	84 c0                                           	test   al,al
   140001398:	74 0d                                           	je     1400013a7 <__tmainCRTStartup+0x253>
   14000139a:	83 7d f4 00                                     	cmp    DWORD PTR [rbp-0xc],0x0
   14000139e:	75 c7                                           	jne    140001367 <__tmainCRTStartup+0x213>
   1400013a0:	eb 05                                           	jmp    1400013a7 <__tmainCRTStartup+0x253>
   1400013a2:	48 83 45 f8 01                                  	add    QWORD PTR [rbp-0x8],0x1
   1400013a7:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   1400013ab:	0f b6 00                                        	movzx  eax,BYTE PTR [rax]
   1400013ae:	84 c0                                           	test   al,al
   1400013b0:	74 0b                                           	je     1400013bd <__tmainCRTStartup+0x269>
   1400013b2:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   1400013b6:	0f b6 00                                        	movzx  eax,BYTE PTR [rax]
   1400013b9:	3c 20                                           	cmp    al,0x20
   1400013bb:	7e e5                                           	jle    1400013a2 <__tmainCRTStartup+0x24e>
   1400013bd:	48 8d 05 fc 7d 00 00                            	lea    rax,[rip+0x7dfc]        # 1400091c0 <__mingw_winmain_lpCmdLine>
   1400013c4:	48 8b 55 f8                                     	mov    rdx,QWORD PTR [rbp-0x8]
   1400013c8:	48 89 10                                        	mov    QWORD PTR [rax],rdx
   1400013cb:	48 8b 05 5e 40 00 00                            	mov    rax,QWORD PTR [rip+0x405e]        # 140005430 <.refptr.__mingw_app_type>
   1400013d2:	8b 00                                           	mov    eax,DWORD PTR [rax]
   1400013d4:	85 c0                                           	test   eax,eax
   1400013d6:	74 21                                           	je     1400013f9 <__tmainCRTStartup+0x2a5>
   1400013d8:	8b 85 7c ff ff ff                               	mov    eax,DWORD PTR [rbp-0x84]
   1400013de:	83 e0 01                                        	and    eax,0x1
   1400013e1:	85 c0                                           	test   eax,eax
   1400013e3:	74 09                                           	je     1400013ee <__tmainCRTStartup+0x29a>
   1400013e5:	0f b7 45 80                                     	movzx  eax,WORD PTR [rbp-0x80]
   1400013e9:	0f b7 c0                                        	movzx  eax,ax
   1400013ec:	eb 05                                           	jmp    1400013f3 <__tmainCRTStartup+0x29f>
   1400013ee:	b8 0a 00 00 00                                  	mov    eax,0xa
   1400013f3:	89 05 07 2c 00 00                               	mov    DWORD PTR [rip+0x2c07],eax        # 140004000 <__data_start__>
   1400013f9:	8b 05 05 7c 00 00                               	mov    eax,DWORD PTR [rip+0x7c05]        # 140009004 <argc>
   1400013ff:	48 8d 15 02 7c 00 00                            	lea    rdx,[rip+0x7c02]        # 140009008 <argv>
   140001406:	89 c1                                           	mov    ecx,eax
   140001408:	e8 76 01 00 00                                  	call   140001583 <duplicate_ppstrings>
   14000140d:	e8 b5 04 00 00                                  	call   1400018c7 <__main>
   140001412:	48 8b 05 f7 3f 00 00                            	mov    rax,QWORD PTR [rip+0x3ff7]        # 140005410 <.refptr.__imp___initenv>
   140001419:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   14000141c:	48 8b 15 ed 7b 00 00                            	mov    rdx,QWORD PTR [rip+0x7bed]        # 140009010 <envp>
   140001423:	48 89 10                                        	mov    QWORD PTR [rax],rdx
   140001426:	48 8b 0d e3 7b 00 00                            	mov    rcx,QWORD PTR [rip+0x7be3]        # 140009010 <envp>
   14000142d:	48 8b 15 d4 7b 00 00                            	mov    rdx,QWORD PTR [rip+0x7bd4]        # 140009008 <argv>
   140001434:	8b 05 ca 7b 00 00                               	mov    eax,DWORD PTR [rip+0x7bca]        # 140009004 <argc>
   14000143a:	49 89 c8                                        	mov    r8,rcx
   14000143d:	89 c1                                           	mov    ecx,eax
   14000143f:	e8 9b 03 00 00                                  	call   1400017df <main>
   140001444:	89 05 d2 7b 00 00                               	mov    DWORD PTR [rip+0x7bd2],eax        # 14000901c <mainret>
   14000144a:	8b 05 d0 7b 00 00                               	mov    eax,DWORD PTR [rip+0x7bd0]        # 140009020 <managedapp>
   140001450:	85 c0                                           	test   eax,eax
   140001452:	75 0d                                           	jne    140001461 <__tmainCRTStartup+0x30d>
   140001454:	8b 05 c2 7b 00 00                               	mov    eax,DWORD PTR [rip+0x7bc2]        # 14000901c <mainret>
   14000145a:	89 c1                                           	mov    ecx,eax
   14000145c:	e8 27 1e 00 00                                  	call   140003288 <exit>
   140001461:	8b 05 bd 7b 00 00                               	mov    eax,DWORD PTR [rip+0x7bbd]        # 140009024 <has_cctor>
   140001467:	85 c0                                           	test   eax,eax
   140001469:	75 05                                           	jne    140001470 <__tmainCRTStartup+0x31c>
   14000146b:	e8 b0 1d 00 00                                  	call   140003220 <_cexit>
   140001470:	8b 05 a6 7b 00 00                               	mov    eax,DWORD PTR [rip+0x7ba6]        # 14000901c <mainret>
   140001476:	48 81 c4 e0 00 00 00                            	add    rsp,0xe0
   14000147d:	5d                                              	pop    rbp
   14000147e:	c3                                              	ret

000000014000147f <check_managed_app>:
   14000147f:	55                                              	push   rbp
   140001480:	48 89 e5                                        	mov    rbp,rsp
   140001483:	48 83 ec 20                                     	sub    rsp,0x20
   140001487:	48 8b 05 b2 3f 00 00                            	mov    rax,QWORD PTR [rip+0x3fb2]        # 140005440 <.refptr.__mingw_initltsdrot_force>
   14000148e:	c7 00 01 00 00 00                               	mov    DWORD PTR [rax],0x1
   140001494:	48 8b 05 b5 3f 00 00                            	mov    rax,QWORD PTR [rip+0x3fb5]        # 140005450 <.refptr.__mingw_initltsdyn_force>
   14000149b:	c7 00 01 00 00 00                               	mov    DWORD PTR [rax],0x1
   1400014a1:	48 8b 05 b8 3f 00 00                            	mov    rax,QWORD PTR [rip+0x3fb8]        # 140005460 <.refptr.__mingw_initltssuo_force>
   1400014a8:	c7 00 01 00 00 00                               	mov    DWORD PTR [rax],0x1
   1400014ae:	48 8b 05 4b 3f 00 00                            	mov    rax,QWORD PTR [rip+0x3f4b]        # 140005400 <.refptr.__image_base__>
   1400014b5:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   1400014b9:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   1400014bd:	0f b7 00                                        	movzx  eax,WORD PTR [rax]
   1400014c0:	66 3d 4d 5a                                     	cmp    ax,0x5a4d
   1400014c4:	74 0a                                           	je     1400014d0 <check_managed_app+0x51>
   1400014c6:	b8 00 00 00 00                                  	mov    eax,0x0
   1400014cb:	e9 ad 00 00 00                                  	jmp    14000157d <check_managed_app+0xfe>
   1400014d0:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   1400014d4:	8b 40 3c                                        	mov    eax,DWORD PTR [rax+0x3c]
   1400014d7:	48 63 d0                                        	movsxd rdx,eax
   1400014da:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   1400014de:	48 01 d0                                        	add    rax,rdx
   1400014e1:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   1400014e5:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400014e9:	8b 00                                           	mov    eax,DWORD PTR [rax]
   1400014eb:	3d 50 45 00 00                                  	cmp    eax,0x4550
   1400014f0:	74 0a                                           	je     1400014fc <check_managed_app+0x7d>
   1400014f2:	b8 00 00 00 00                                  	mov    eax,0x0
   1400014f7:	e9 81 00 00 00                                  	jmp    14000157d <check_managed_app+0xfe>
   1400014fc:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140001500:	48 83 c0 18                                     	add    rax,0x18
   140001504:	48 89 45 e8                                     	mov    QWORD PTR [rbp-0x18],rax
   140001508:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   14000150c:	0f b7 00                                        	movzx  eax,WORD PTR [rax]
   14000150f:	0f b7 c0                                        	movzx  eax,ax
   140001512:	3d 0b 01 00 00                                  	cmp    eax,0x10b
   140001517:	74 09                                           	je     140001522 <check_managed_app+0xa3>
   140001519:	3d 0b 02 00 00                                  	cmp    eax,0x20b
   14000151e:	74 29                                           	je     140001549 <check_managed_app+0xca>
   140001520:	eb 56                                           	jmp    140001578 <check_managed_app+0xf9>
   140001522:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140001526:	8b 40 5c                                        	mov    eax,DWORD PTR [rax+0x5c]
   140001529:	83 f8 0e                                        	cmp    eax,0xe
   14000152c:	77 07                                           	ja     140001535 <check_managed_app+0xb6>
   14000152e:	b8 00 00 00 00                                  	mov    eax,0x0
   140001533:	eb 48                                           	jmp    14000157d <check_managed_app+0xfe>
   140001535:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140001539:	8b 80 d0 00 00 00                               	mov    eax,DWORD PTR [rax+0xd0]
   14000153f:	85 c0                                           	test   eax,eax
   140001541:	0f 95 c0                                        	setne  al
   140001544:	0f b6 c0                                        	movzx  eax,al
   140001547:	eb 34                                           	jmp    14000157d <check_managed_app+0xfe>
   140001549:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   14000154d:	48 89 45 e0                                     	mov    QWORD PTR [rbp-0x20],rax
   140001551:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140001555:	8b 40 6c                                        	mov    eax,DWORD PTR [rax+0x6c]
   140001558:	83 f8 0e                                        	cmp    eax,0xe
   14000155b:	77 07                                           	ja     140001564 <check_managed_app+0xe5>
   14000155d:	b8 00 00 00 00                                  	mov    eax,0x0
   140001562:	eb 19                                           	jmp    14000157d <check_managed_app+0xfe>
   140001564:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140001568:	8b 80 e0 00 00 00                               	mov    eax,DWORD PTR [rax+0xe0]
   14000156e:	85 c0                                           	test   eax,eax
   140001570:	0f 95 c0                                        	setne  al
   140001573:	0f b6 c0                                        	movzx  eax,al
   140001576:	eb 05                                           	jmp    14000157d <check_managed_app+0xfe>
   140001578:	b8 00 00 00 00                                  	mov    eax,0x0
   14000157d:	48 83 c4 20                                     	add    rsp,0x20
   140001581:	5d                                              	pop    rbp
   140001582:	c3                                              	ret

0000000140001583 <duplicate_ppstrings>:
   140001583:	55                                              	push   rbp
   140001584:	53                                              	push   rbx
   140001585:	48 83 ec 48                                     	sub    rsp,0x48
   140001589:	48 8d 6c 24 40                                  	lea    rbp,[rsp+0x40]
   14000158e:	89 4d 20                                        	mov    DWORD PTR [rbp+0x20],ecx
   140001591:	48 89 55 28                                     	mov    QWORD PTR [rbp+0x28],rdx
   140001595:	8b 45 20                                        	mov    eax,DWORD PTR [rbp+0x20]
   140001598:	83 c0 01                                        	add    eax,0x1
   14000159b:	48 98                                           	cdqe
   14000159d:	48 c1 e0 03                                     	shl    rax,0x3
   1400015a1:	48 89 c1                                        	mov    rcx,rax
   1400015a4:	e8 f7 1c 00 00                                  	call   1400032a0 <malloc>
   1400015a9:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   1400015ad:	48 8b 45 28                                     	mov    rax,QWORD PTR [rbp+0x28]
   1400015b1:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   1400015b4:	48 89 45 e8                                     	mov    QWORD PTR [rbp-0x18],rax
   1400015b8:	c7 45 fc 00 00 00 00                            	mov    DWORD PTR [rbp-0x4],0x0
   1400015bf:	e9 8c 00 00 00                                  	jmp    140001650 <duplicate_ppstrings+0xcd>
   1400015c4:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   1400015c7:	48 98                                           	cdqe
   1400015c9:	48 8d 14 c5 00 00 00 00                         	lea    rdx,[rax*8+0x0]
   1400015d1:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   1400015d5:	48 01 d0                                        	add    rax,rdx
   1400015d8:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   1400015db:	48 89 c1                                        	mov    rcx,rax
   1400015de:	e8 dd 1c 00 00                                  	call   1400032c0 <strlen>
   1400015e3:	48 83 c0 01                                     	add    rax,0x1
   1400015e7:	48 89 45 e0                                     	mov    QWORD PTR [rbp-0x20],rax
   1400015eb:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   1400015ee:	48 98                                           	cdqe
   1400015f0:	48 8d 14 c5 00 00 00 00                         	lea    rdx,[rax*8+0x0]
   1400015f8:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400015fc:	48 8d 1c 02                                     	lea    rbx,[rdx+rax*1]
   140001600:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140001604:	48 89 c1                                        	mov    rcx,rax
   140001607:	e8 94 1c 00 00                                  	call   1400032a0 <malloc>
   14000160c:	48 89 03                                        	mov    QWORD PTR [rbx],rax
   14000160f:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001612:	48 98                                           	cdqe
   140001614:	48 8d 14 c5 00 00 00 00                         	lea    rdx,[rax*8+0x0]
   14000161c:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140001620:	48 01 d0                                        	add    rax,rdx
   140001623:	48 8b 10                                        	mov    rdx,QWORD PTR [rax]
   140001626:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001629:	48 98                                           	cdqe
   14000162b:	48 8d 0c c5 00 00 00 00                         	lea    rcx,[rax*8+0x0]
   140001633:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140001637:	48 01 c8                                        	add    rax,rcx
   14000163a:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   14000163d:	48 8b 4d e0                                     	mov    rcx,QWORD PTR [rbp-0x20]
   140001641:	49 89 c8                                        	mov    r8,rcx
   140001644:	48 89 c1                                        	mov    rcx,rax
   140001647:	e8 5c 1c 00 00                                  	call   1400032a8 <memcpy>
   14000164c:	83 45 fc 01                                     	add    DWORD PTR [rbp-0x4],0x1
   140001650:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001653:	3b 45 20                                        	cmp    eax,DWORD PTR [rbp+0x20]
   140001656:	0f 8c 68 ff ff ff                               	jl     1400015c4 <duplicate_ppstrings+0x41>
   14000165c:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   14000165f:	48 98                                           	cdqe
   140001661:	48 8d 14 c5 00 00 00 00                         	lea    rdx,[rax*8+0x0]
   140001669:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   14000166d:	48 01 d0                                        	add    rax,rdx
   140001670:	48 c7 00 00 00 00 00                            	mov    QWORD PTR [rax],0x0
   140001677:	48 8b 45 28                                     	mov    rax,QWORD PTR [rbp+0x28]
   14000167b:	48 8b 55 f0                                     	mov    rdx,QWORD PTR [rbp-0x10]
   14000167f:	48 89 10                                        	mov    QWORD PTR [rax],rdx
   140001682:	90                                              	nop
   140001683:	48 83 c4 48                                     	add    rsp,0x48
   140001687:	5b                                              	pop    rbx
   140001688:	5d                                              	pop    rbp
   140001689:	c3                                              	ret

000000014000168a <atexit>:
   14000168a:	55                                              	push   rbp
   14000168b:	48 89 e5                                        	mov    rbp,rsp
   14000168e:	48 83 ec 20                                     	sub    rsp,0x20
   140001692:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140001696:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   14000169a:	48 89 c1                                        	mov    rcx,rax
   14000169d:	e8 ae 19 00 00                                  	call   140003050 <_onexit>
   1400016a2:	48 85 c0                                        	test   rax,rax
   1400016a5:	74 07                                           	je     1400016ae <atexit+0x24>
   1400016a7:	b8 00 00 00 00                                  	mov    eax,0x0
   1400016ac:	eb 05                                           	jmp    1400016b3 <atexit+0x29>
   1400016ae:	b8 ff ff ff ff                                  	mov    eax,0xffffffff
   1400016b3:	48 83 c4 20                                     	add    rsp,0x20
   1400016b7:	5d                                              	pop    rbp
   1400016b8:	c3                                              	ret
   1400016b9:	90                                              	nop
   1400016ba:	90                                              	nop
   1400016bb:	90                                              	nop
   1400016bc:	90                                              	nop
   1400016bd:	90                                              	nop
   1400016be:	90                                              	nop
   1400016bf:	90                                              	nop

00000001400016c0 <.weak.__register_frame_info.hmod_libgcc>:
   1400016c0:	c3                                              	ret
   1400016c1:	66 66 2e 0f 1f 84 00 00 00 00 00                	data16 cs nop WORD PTR [rax+rax*1+0x0]
   1400016cc:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

00000001400016d0 <.weak.__deregister_frame_info.hmod_libgcc>:
   1400016d0:	31 c0                                           	xor    eax,eax
   1400016d2:	c3                                              	ret
   1400016d3:	66 66 2e 0f 1f 84 00 00 00 00 00                	data16 cs nop WORD PTR [rax+rax*1+0x0]
   1400016de:	66 90                                           	xchg   ax,ax

00000001400016e0 <__gcc_register_frame>:
   1400016e0:	55                                              	push   rbp
   1400016e1:	57                                              	push   rdi
   1400016e2:	56                                              	push   rsi
   1400016e3:	53                                              	push   rbx
   1400016e4:	48 83 ec 28                                     	sub    rsp,0x28
   1400016e8:	48 8d 6c 24 20                                  	lea    rbp,[rsp+0x20]
   1400016ed:	48 8d 35 0c 39 00 00                            	lea    rsi,[rip+0x390c]        # 140005000 <.rdata>
   1400016f4:	48 89 f1                                        	mov    rcx,rsi
   1400016f7:	ff 15 23 8b 00 00                               	call   QWORD PTR [rip+0x8b23]        # 14000a220 <__imp_GetModuleHandleA>
   1400016fd:	48 89 c3                                        	mov    rbx,rax
   140001700:	48 85 c0                                        	test   rax,rax
   140001703:	74 6b                                           	je     140001770 <__gcc_register_frame+0x90>
   140001705:	48 89 f1                                        	mov    rcx,rsi
   140001708:	ff 15 3a 8b 00 00                               	call   QWORD PTR [rip+0x8b3a]        # 14000a248 <__imp_LoadLibraryA>
   14000170e:	48 8b 3d 13 8b 00 00                            	mov    rdi,QWORD PTR [rip+0x8b13]        # 14000a228 <__imp_GetProcAddress>
   140001715:	48 8d 15 f7 38 00 00                            	lea    rdx,[rip+0x38f7]        # 140005013 <.rdata+0x13>
   14000171c:	48 89 d9                                        	mov    rcx,rbx
   14000171f:	48 89 05 1a 79 00 00                            	mov    QWORD PTR [rip+0x791a],rax        # 140009040 <hmod_libgcc>
   140001726:	ff d7                                           	call   rdi
   140001728:	48 8d 15 fa 38 00 00                            	lea    rdx,[rip+0x38fa]        # 140005029 <.rdata+0x29>
   14000172f:	48 89 d9                                        	mov    rcx,rbx
   140001732:	48 89 c6                                        	mov    rsi,rax
   140001735:	ff d7                                           	call   rdi
   140001737:	48 89 05 d2 28 00 00                            	mov    QWORD PTR [rip+0x28d2],rax        # 140004010 <deregister_frame_fn>
   14000173e:	48 85 f6                                        	test   rsi,rsi
   140001741:	74 10                                           	je     140001753 <__gcc_register_frame+0x73>
   140001743:	48 8d 15 16 79 00 00                            	lea    rdx,[rip+0x7916]        # 140009060 <obj>
   14000174a:	48 8d 0d af 48 00 00                            	lea    rcx,[rip+0x48af]        # 140006000 <__EH_FRAME_BEGIN__>
   140001751:	ff d6                                           	call   rsi
   140001753:	48 8d 0d 36 00 00 00                            	lea    rcx,[rip+0x36]        # 140001790 <__gcc_deregister_frame>
   14000175a:	48 83 c4 28                                     	add    rsp,0x28
   14000175e:	5b                                              	pop    rbx
   14000175f:	5e                                              	pop    rsi
   140001760:	5f                                              	pop    rdi
   140001761:	5d                                              	pop    rbp
   140001762:	e9 23 ff ff ff                                  	jmp    14000168a <atexit>
   140001767:	66 0f 1f 84 00 00 00 00 00                      	nop    WORD PTR [rax+rax*1+0x0]
   140001770:	48 8d 05 59 ff ff ff                            	lea    rax,[rip+0xffffffffffffff59]        # 1400016d0 <.weak.__deregister_frame_info.hmod_libgcc>
   140001777:	48 8d 35 42 ff ff ff                            	lea    rsi,[rip+0xffffffffffffff42]        # 1400016c0 <.weak.__register_frame_info.hmod_libgcc>
   14000177e:	48 89 05 8b 28 00 00                            	mov    QWORD PTR [rip+0x288b],rax        # 140004010 <deregister_frame_fn>
   140001785:	eb bc                                           	jmp    140001743 <__gcc_register_frame+0x63>
   140001787:	66 0f 1f 84 00 00 00 00 00                      	nop    WORD PTR [rax+rax*1+0x0]

0000000140001790 <__gcc_deregister_frame>:
   140001790:	55                                              	push   rbp
   140001791:	48 89 e5                                        	mov    rbp,rsp
   140001794:	48 83 ec 20                                     	sub    rsp,0x20
   140001798:	48 8b 05 71 28 00 00                            	mov    rax,QWORD PTR [rip+0x2871]        # 140004010 <deregister_frame_fn>
   14000179f:	48 85 c0                                        	test   rax,rax
   1400017a2:	74 09                                           	je     1400017ad <__gcc_deregister_frame+0x1d>
   1400017a4:	48 8d 0d 55 48 00 00                            	lea    rcx,[rip+0x4855]        # 140006000 <__EH_FRAME_BEGIN__>
   1400017ab:	ff d0                                           	call   rax
   1400017ad:	48 8b 0d 8c 78 00 00                            	mov    rcx,QWORD PTR [rip+0x788c]        # 140009040 <hmod_libgcc>
   1400017b4:	48 85 c9                                        	test   rcx,rcx
   1400017b7:	74 0f                                           	je     1400017c8 <__gcc_deregister_frame+0x38>
   1400017b9:	48 83 c4 20                                     	add    rsp,0x20
   1400017bd:	5d                                              	pop    rbp
   1400017be:	48 ff 25 4b 8a 00 00                            	rex.W jmp QWORD PTR [rip+0x8a4b]        # 14000a210 <__imp_FreeLibrary>
   1400017c5:	0f 1f 00                                        	nop    DWORD PTR [rax]
   1400017c8:	48 83 c4 20                                     	add    rsp,0x20
   1400017cc:	5d                                              	pop    rbp
   1400017cd:	c3                                              	ret
   1400017ce:	90                                              	nop
   1400017cf:	90                                              	nop

00000001400017d0 <square(int)>:
square(int):
C:/Users/quist/AppData/Local/Temp/compiler-explorer-compiler2023210-13636-snhmpj.zf8hn/example.cpp:2
   1400017d0:	55                                              	push   rbp
   1400017d1:	48 89 e5                                        	mov    rbp,rsp
   1400017d4:	89 4d 10                                        	mov    DWORD PTR [rbp+0x10],ecx
C:/Users/quist/AppData/Local/Temp/compiler-explorer-compiler2023210-13636-snhmpj.zf8hn/example.cpp:3
   1400017d7:	8b 45 10                                        	mov    eax,DWORD PTR [rbp+0x10]
   1400017da:	0f af c0                                        	imul   eax,eax
C:/Users/quist/AppData/Local/Temp/compiler-explorer-compiler2023210-13636-snhmpj.zf8hn/example.cpp:4
   1400017dd:	5d                                              	pop    rbp
   1400017de:	c3                                              	ret

00000001400017df <main>:
main():
C:/Users/quist/AppData/Local/Temp/compiler-explorer-compiler2023210-13636-snhmpj.zf8hn/example.cpp:6
   1400017df:	55                                              	push   rbp
   1400017e0:	48 89 e5                                        	mov    rbp,rsp
   1400017e3:	48 83 ec 20                                     	sub    rsp,0x20
   1400017e7:	89 4d 10                                        	mov    DWORD PTR [rbp+0x10],ecx
   1400017ea:	48 89 55 18                                     	mov    QWORD PTR [rbp+0x18],rdx
   1400017ee:	e8 d4 00 00 00                                  	call   1400018c7 <__main>
C:/Users/quist/AppData/Local/Temp/compiler-explorer-compiler2023210-13636-snhmpj.zf8hn/example.cpp:7
   1400017f3:	8b 45 10                                        	mov    eax,DWORD PTR [rbp+0x10]
   1400017f6:	83 c0 01                                        	add    eax,0x1
   1400017f9:	89 c1                                           	mov    ecx,eax
   1400017fb:	e8 d0 ff ff ff                                  	call   1400017d0 <square(int)>
   140001800:	90                                              	nop
C:/Users/quist/AppData/Local/Temp/compiler-explorer-compiler2023210-13636-snhmpj.zf8hn/example.cpp:8
   140001801:	48 83 c4 20                                     	add    rsp,0x20
   140001805:	5d                                              	pop    rbp
   140001806:	c3                                              	ret
   140001807:	90                                              	nop
   140001808:	90                                              	nop
   140001809:	90                                              	nop
   14000180a:	90                                              	nop
   14000180b:	90                                              	nop
   14000180c:	90                                              	nop
   14000180d:	90                                              	nop
   14000180e:	90                                              	nop
   14000180f:	90                                              	nop

0000000140001810 <__do_global_dtors>:
   140001810:	55                                              	push   rbp
   140001811:	48 89 e5                                        	mov    rbp,rsp
   140001814:	48 83 ec 20                                     	sub    rsp,0x20
   140001818:	eb 1e                                           	jmp    140001838 <__do_global_dtors+0x28>
   14000181a:	48 8b 05 ff 27 00 00                            	mov    rax,QWORD PTR [rip+0x27ff]        # 140004020 <p.0>
   140001821:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   140001824:	ff d0                                           	call   rax
   140001826:	48 8b 05 f3 27 00 00                            	mov    rax,QWORD PTR [rip+0x27f3]        # 140004020 <p.0>
   14000182d:	48 83 c0 08                                     	add    rax,0x8
   140001831:	48 89 05 e8 27 00 00                            	mov    QWORD PTR [rip+0x27e8],rax        # 140004020 <p.0>
   140001838:	48 8b 05 e1 27 00 00                            	mov    rax,QWORD PTR [rip+0x27e1]        # 140004020 <p.0>
   14000183f:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   140001842:	48 85 c0                                        	test   rax,rax
   140001845:	75 d3                                           	jne    14000181a <__do_global_dtors+0xa>
   140001847:	90                                              	nop
   140001848:	90                                              	nop
   140001849:	48 83 c4 20                                     	add    rsp,0x20
   14000184d:	5d                                              	pop    rbp
   14000184e:	c3                                              	ret

000000014000184f <__do_global_ctors>:
   14000184f:	55                                              	push   rbp
   140001850:	48 89 e5                                        	mov    rbp,rsp
   140001853:	48 83 ec 30                                     	sub    rsp,0x30
   140001857:	48 8b 05 52 3b 00 00                            	mov    rax,QWORD PTR [rip+0x3b52]        # 1400053b0 <.refptr.__CTOR_LIST__>
   14000185e:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   140001861:	89 45 fc                                        	mov    DWORD PTR [rbp-0x4],eax
   140001864:	83 7d fc ff                                     	cmp    DWORD PTR [rbp-0x4],0xffffffff
   140001868:	75 25                                           	jne    14000188f <__do_global_ctors+0x40>
   14000186a:	c7 45 fc 00 00 00 00                            	mov    DWORD PTR [rbp-0x4],0x0
   140001871:	eb 04                                           	jmp    140001877 <__do_global_ctors+0x28>
   140001873:	83 45 fc 01                                     	add    DWORD PTR [rbp-0x4],0x1
   140001877:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   14000187a:	8d 50 01                                        	lea    edx,[rax+0x1]
   14000187d:	48 8b 05 2c 3b 00 00                            	mov    rax,QWORD PTR [rip+0x3b2c]        # 1400053b0 <.refptr.__CTOR_LIST__>
   140001884:	89 d2                                           	mov    edx,edx
   140001886:	48 8b 04 d0                                     	mov    rax,QWORD PTR [rax+rdx*8]
   14000188a:	48 85 c0                                        	test   rax,rax
   14000188d:	75 e4                                           	jne    140001873 <__do_global_ctors+0x24>
   14000188f:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001892:	89 45 f8                                        	mov    DWORD PTR [rbp-0x8],eax
   140001895:	eb 14                                           	jmp    1400018ab <__do_global_ctors+0x5c>
   140001897:	48 8b 05 12 3b 00 00                            	mov    rax,QWORD PTR [rip+0x3b12]        # 1400053b0 <.refptr.__CTOR_LIST__>
   14000189e:	8b 55 f8                                        	mov    edx,DWORD PTR [rbp-0x8]
   1400018a1:	48 8b 04 d0                                     	mov    rax,QWORD PTR [rax+rdx*8]
   1400018a5:	ff d0                                           	call   rax
   1400018a7:	83 6d f8 01                                     	sub    DWORD PTR [rbp-0x8],0x1
   1400018ab:	83 7d f8 00                                     	cmp    DWORD PTR [rbp-0x8],0x0
   1400018af:	75 e6                                           	jne    140001897 <__do_global_ctors+0x48>
   1400018b1:	48 8d 05 58 ff ff ff                            	lea    rax,[rip+0xffffffffffffff58]        # 140001810 <__do_global_dtors>
   1400018b8:	48 89 c1                                        	mov    rcx,rax
   1400018bb:	e8 ca fd ff ff                                  	call   14000168a <atexit>
   1400018c0:	90                                              	nop
   1400018c1:	48 83 c4 30                                     	add    rsp,0x30
   1400018c5:	5d                                              	pop    rbp
   1400018c6:	c3                                              	ret

00000001400018c7 <__main>:
   1400018c7:	55                                              	push   rbp
   1400018c8:	48 89 e5                                        	mov    rbp,rsp
   1400018cb:	48 83 ec 20                                     	sub    rsp,0x20
   1400018cf:	8b 05 cb 77 00 00                               	mov    eax,DWORD PTR [rip+0x77cb]        # 1400090a0 <initialized>
   1400018d5:	85 c0                                           	test   eax,eax
   1400018d7:	75 0f                                           	jne    1400018e8 <__main+0x21>
   1400018d9:	c7 05 bd 77 00 00 01 00 00 00                   	mov    DWORD PTR [rip+0x77bd],0x1        # 1400090a0 <initialized>
   1400018e3:	e8 67 ff ff ff                                  	call   14000184f <__do_global_ctors>
   1400018e8:	90                                              	nop
   1400018e9:	48 83 c4 20                                     	add    rsp,0x20
   1400018ed:	5d                                              	pop    rbp
   1400018ee:	c3                                              	ret
   1400018ef:	90                                              	nop

00000001400018f0 <_setargv>:
   1400018f0:	55                                              	push   rbp
   1400018f1:	48 89 e5                                        	mov    rbp,rsp
   1400018f4:	b8 00 00 00 00                                  	mov    eax,0x0
   1400018f9:	5d                                              	pop    rbp
   1400018fa:	c3                                              	ret
   1400018fb:	90                                              	nop
   1400018fc:	90                                              	nop
   1400018fd:	90                                              	nop
   1400018fe:	90                                              	nop
   1400018ff:	90                                              	nop

0000000140001900 <__dyn_tls_init>:
   140001900:	55                                              	push   rbp
   140001901:	48 89 e5                                        	mov    rbp,rsp
   140001904:	48 83 ec 30                                     	sub    rsp,0x30
   140001908:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   14000190c:	89 55 18                                        	mov    DWORD PTR [rbp+0x18],edx
   14000190f:	4c 89 45 20                                     	mov    QWORD PTR [rbp+0x20],r8
   140001913:	48 8b 05 76 3a 00 00                            	mov    rax,QWORD PTR [rip+0x3a76]        # 140005390 <.refptr._CRT_MT>
   14000191a:	8b 00                                           	mov    eax,DWORD PTR [rax]
   14000191c:	83 f8 02                                        	cmp    eax,0x2
   14000191f:	74 0d                                           	je     14000192e <__dyn_tls_init+0x2e>
   140001921:	48 8b 05 68 3a 00 00                            	mov    rax,QWORD PTR [rip+0x3a68]        # 140005390 <.refptr._CRT_MT>
   140001928:	c7 00 02 00 00 00                               	mov    DWORD PTR [rax],0x2
   14000192e:	83 7d 18 02                                     	cmp    DWORD PTR [rbp+0x18],0x2
   140001932:	74 23                                           	je     140001957 <__dyn_tls_init+0x57>
   140001934:	83 7d 18 01                                     	cmp    DWORD PTR [rbp+0x18],0x1
   140001938:	75 16                                           	jne    140001950 <__dyn_tls_init+0x50>
   14000193a:	48 8b 4d 20                                     	mov    rcx,QWORD PTR [rbp+0x20]
   14000193e:	8b 55 18                                        	mov    edx,DWORD PTR [rbp+0x18]
   140001941:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140001945:	49 89 c8                                        	mov    r8,rcx
   140001948:	48 89 c1                                        	mov    rcx,rax
   14000194b:	e8 61 0f 00 00                                  	call   1400028b1 <__mingw_TLScallback>
   140001950:	b8 01 00 00 00                                  	mov    eax,0x1
   140001955:	eb 46                                           	jmp    14000199d <__dyn_tls_init+0x9d>
   140001957:	48 8d 05 f2 96 00 00                            	lea    rax,[rip+0x96f2]        # 14000b050 <___crt_xp_end__>
   14000195e:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140001962:	48 83 45 f8 08                                  	add    QWORD PTR [rbp-0x8],0x8
   140001967:	eb 22                                           	jmp    14000198b <__dyn_tls_init+0x8b>
   140001969:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   14000196d:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   140001971:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140001975:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   140001978:	48 85 c0                                        	test   rax,rax
   14000197b:	74 09                                           	je     140001986 <__dyn_tls_init+0x86>
   14000197d:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140001981:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   140001984:	ff d0                                           	call   rax
   140001986:	48 83 45 f8 08                                  	add    QWORD PTR [rbp-0x8],0x8
   14000198b:	48 8d 05 c6 96 00 00                            	lea    rax,[rip+0x96c6]        # 14000b058 <__xd_z>
   140001992:	48 39 45 f8                                     	cmp    QWORD PTR [rbp-0x8],rax
   140001996:	75 d1                                           	jne    140001969 <__dyn_tls_init+0x69>
   140001998:	b8 01 00 00 00                                  	mov    eax,0x1
   14000199d:	48 83 c4 30                                     	add    rsp,0x30
   1400019a1:	5d                                              	pop    rbp
   1400019a2:	c3                                              	ret

00000001400019a3 <__tlregdtor>:
   1400019a3:	55                                              	push   rbp
   1400019a4:	48 89 e5                                        	mov    rbp,rsp
   1400019a7:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   1400019ab:	48 83 7d 10 00                                  	cmp    QWORD PTR [rbp+0x10],0x0
   1400019b0:	75 07                                           	jne    1400019b9 <__tlregdtor+0x16>
   1400019b2:	b8 00 00 00 00                                  	mov    eax,0x0
   1400019b7:	eb 05                                           	jmp    1400019be <__tlregdtor+0x1b>
   1400019b9:	b8 00 00 00 00                                  	mov    eax,0x0
   1400019be:	5d                                              	pop    rbp
   1400019bf:	c3                                              	ret

00000001400019c0 <__dyn_tls_dtor>:
   1400019c0:	55                                              	push   rbp
   1400019c1:	48 89 e5                                        	mov    rbp,rsp
   1400019c4:	48 83 ec 20                                     	sub    rsp,0x20
   1400019c8:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   1400019cc:	89 55 18                                        	mov    DWORD PTR [rbp+0x18],edx
   1400019cf:	4c 89 45 20                                     	mov    QWORD PTR [rbp+0x20],r8
   1400019d3:	83 7d 18 03                                     	cmp    DWORD PTR [rbp+0x18],0x3
   1400019d7:	74 0d                                           	je     1400019e6 <__dyn_tls_dtor+0x26>
   1400019d9:	83 7d 18 00                                     	cmp    DWORD PTR [rbp+0x18],0x0
   1400019dd:	74 07                                           	je     1400019e6 <__dyn_tls_dtor+0x26>
   1400019df:	b8 01 00 00 00                                  	mov    eax,0x1
   1400019e4:	eb 1b                                           	jmp    140001a01 <__dyn_tls_dtor+0x41>
   1400019e6:	48 8b 4d 20                                     	mov    rcx,QWORD PTR [rbp+0x20]
   1400019ea:	8b 55 18                                        	mov    edx,DWORD PTR [rbp+0x18]
   1400019ed:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   1400019f1:	49 89 c8                                        	mov    r8,rcx
   1400019f4:	48 89 c1                                        	mov    rcx,rax
   1400019f7:	e8 b5 0e 00 00                                  	call   1400028b1 <__mingw_TLScallback>
   1400019fc:	b8 01 00 00 00                                  	mov    eax,0x1
   140001a01:	48 83 c4 20                                     	add    rsp,0x20
   140001a05:	5d                                              	pop    rbp
   140001a06:	c3                                              	ret
   140001a07:	90                                              	nop
   140001a08:	90                                              	nop
   140001a09:	90                                              	nop
   140001a0a:	90                                              	nop
   140001a0b:	90                                              	nop
   140001a0c:	90                                              	nop
   140001a0d:	90                                              	nop
   140001a0e:	90                                              	nop
   140001a0f:	90                                              	nop

0000000140001a10 <_matherr>:
   140001a10:	55                                              	push   rbp
   140001a11:	53                                              	push   rbx
   140001a12:	48 81 ec 88 00 00 00                            	sub    rsp,0x88
   140001a19:	48 8d 6c 24 50                                  	lea    rbp,[rsp+0x50]
   140001a1e:	0f 29 75 00                                     	movaps XMMWORD PTR [rbp+0x0],xmm6
   140001a22:	0f 29 7d 10                                     	movaps XMMWORD PTR [rbp+0x10],xmm7
   140001a26:	44 0f 29 45 20                                  	movaps XMMWORD PTR [rbp+0x20],xmm8
   140001a2b:	48 89 4d 50                                     	mov    QWORD PTR [rbp+0x50],rcx
   140001a2f:	48 8b 45 50                                     	mov    rax,QWORD PTR [rbp+0x50]
   140001a33:	8b 00                                           	mov    eax,DWORD PTR [rax]
   140001a35:	83 f8 06                                        	cmp    eax,0x6
   140001a38:	77 70                                           	ja     140001aaa <_matherr+0x9a>
   140001a3a:	89 c0                                           	mov    eax,eax
   140001a3c:	48 8d 14 85 00 00 00 00                         	lea    rdx,[rax*4+0x0]
   140001a44:	48 8d 05 79 37 00 00                            	lea    rax,[rip+0x3779]        # 1400051c4 <.rdata+0x124>
   140001a4b:	8b 04 02                                        	mov    eax,DWORD PTR [rdx+rax*1]
   140001a4e:	48 98                                           	cdqe
   140001a50:	48 8d 15 6d 37 00 00                            	lea    rdx,[rip+0x376d]        # 1400051c4 <.rdata+0x124>
   140001a57:	48 01 d0                                        	add    rax,rdx
   140001a5a:	ff e0                                           	jmp    rax
   140001a5c:	48 8d 05 3d 36 00 00                            	lea    rax,[rip+0x363d]        # 1400050a0 <.rdata>
   140001a63:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140001a67:	eb 4d                                           	jmp    140001ab6 <_matherr+0xa6>
   140001a69:	48 8d 05 4f 36 00 00                            	lea    rax,[rip+0x364f]        # 1400050bf <.rdata+0x1f>
   140001a70:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140001a74:	eb 40                                           	jmp    140001ab6 <_matherr+0xa6>
   140001a76:	48 8d 05 63 36 00 00                            	lea    rax,[rip+0x3663]        # 1400050e0 <.rdata+0x40>
   140001a7d:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140001a81:	eb 33                                           	jmp    140001ab6 <_matherr+0xa6>
   140001a83:	48 8d 05 76 36 00 00                            	lea    rax,[rip+0x3676]        # 140005100 <.rdata+0x60>
   140001a8a:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140001a8e:	eb 26                                           	jmp    140001ab6 <_matherr+0xa6>
   140001a90:	48 8d 05 91 36 00 00                            	lea    rax,[rip+0x3691]        # 140005128 <.rdata+0x88>
   140001a97:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140001a9b:	eb 19                                           	jmp    140001ab6 <_matherr+0xa6>
   140001a9d:	48 8d 05 ac 36 00 00                            	lea    rax,[rip+0x36ac]        # 140005150 <.rdata+0xb0>
   140001aa4:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140001aa8:	eb 0c                                           	jmp    140001ab6 <_matherr+0xa6>
   140001aaa:	48 8d 05 d5 36 00 00                            	lea    rax,[rip+0x36d5]        # 140005186 <.rdata+0xe6>
   140001ab1:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140001ab5:	90                                              	nop
   140001ab6:	48 8b 45 50                                     	mov    rax,QWORD PTR [rbp+0x50]
   140001aba:	f2 44 0f 10 40 20                               	movsd  xmm8,QWORD PTR [rax+0x20]
   140001ac0:	48 8b 45 50                                     	mov    rax,QWORD PTR [rbp+0x50]
   140001ac4:	f2 0f 10 78 18                                  	movsd  xmm7,QWORD PTR [rax+0x18]
   140001ac9:	48 8b 45 50                                     	mov    rax,QWORD PTR [rbp+0x50]
   140001acd:	f2 0f 10 70 10                                  	movsd  xmm6,QWORD PTR [rax+0x10]
   140001ad2:	48 8b 45 50                                     	mov    rax,QWORD PTR [rbp+0x50]
   140001ad6:	48 8b 58 08                                     	mov    rbx,QWORD PTR [rax+0x8]
   140001ada:	b9 02 00 00 00                                  	mov    ecx,0x2
   140001adf:	e8 c4 16 00 00                                  	call   1400031a8 <__acrt_iob_func>
   140001ae4:	48 89 c1                                        	mov    rcx,rax
   140001ae7:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140001aeb:	f2 44 0f 11 44 24 30                            	movsd  QWORD PTR [rsp+0x30],xmm8
   140001af2:	f2 0f 11 7c 24 28                               	movsd  QWORD PTR [rsp+0x28],xmm7
   140001af8:	f2 0f 11 74 24 20                               	movsd  QWORD PTR [rsp+0x20],xmm6
   140001afe:	49 89 d9                                        	mov    r9,rbx
   140001b01:	49 89 c0                                        	mov    r8,rax
   140001b04:	48 8d 05 8d 36 00 00                            	lea    rax,[rip+0x368d]        # 140005198 <.rdata+0xf8>
   140001b0b:	48 89 c2                                        	mov    rdx,rax
   140001b0e:	e8 dd 13 00 00                                  	call   140002ef0 <fprintf>
   140001b13:	b8 00 00 00 00                                  	mov    eax,0x0
   140001b18:	0f 28 75 00                                     	movaps xmm6,XMMWORD PTR [rbp+0x0]
   140001b1c:	0f 28 7d 10                                     	movaps xmm7,XMMWORD PTR [rbp+0x10]
   140001b20:	44 0f 28 45 20                                  	movaps xmm8,XMMWORD PTR [rbp+0x20]
   140001b25:	48 81 c4 88 00 00 00                            	add    rsp,0x88
   140001b2c:	5b                                              	pop    rbx
   140001b2d:	5d                                              	pop    rbp
   140001b2e:	c3                                              	ret
   140001b2f:	90                                              	nop

0000000140001b30 <_fpreset>:
   140001b30:	55                                              	push   rbp
   140001b31:	48 89 e5                                        	mov    rbp,rsp
   140001b34:	db e3                                           	fninit
   140001b36:	90                                              	nop
   140001b37:	5d                                              	pop    rbp
   140001b38:	c3                                              	ret
   140001b39:	90                                              	nop
   140001b3a:	90                                              	nop
   140001b3b:	90                                              	nop
   140001b3c:	90                                              	nop
   140001b3d:	90                                              	nop
   140001b3e:	90                                              	nop
   140001b3f:	90                                              	nop

0000000140001b40 <__report_error>:
   140001b40:	55                                              	push   rbp
   140001b41:	53                                              	push   rbx
   140001b42:	48 83 ec 38                                     	sub    rsp,0x38
   140001b46:	48 8d 6c 24 30                                  	lea    rbp,[rsp+0x30]
   140001b4b:	48 89 4d 20                                     	mov    QWORD PTR [rbp+0x20],rcx
   140001b4f:	48 89 55 28                                     	mov    QWORD PTR [rbp+0x28],rdx
   140001b53:	4c 89 45 30                                     	mov    QWORD PTR [rbp+0x30],r8
   140001b57:	4c 89 4d 38                                     	mov    QWORD PTR [rbp+0x38],r9
   140001b5b:	48 8d 45 28                                     	lea    rax,[rbp+0x28]
   140001b5f:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140001b63:	b9 02 00 00 00                                  	mov    ecx,0x2
   140001b68:	e8 3b 16 00 00                                  	call   1400031a8 <__acrt_iob_func>
   140001b6d:	49 89 c1                                        	mov    r9,rax
   140001b70:	41 b8 1b 00 00 00                               	mov    r8d,0x1b
   140001b76:	ba 01 00 00 00                                  	mov    edx,0x1
   140001b7b:	48 8d 05 5e 36 00 00                            	lea    rax,[rip+0x365e]        # 1400051e0 <.rdata>
   140001b82:	48 89 c1                                        	mov    rcx,rax
   140001b85:	e8 0e 17 00 00                                  	call   140003298 <fwrite>
   140001b8a:	48 8b 5d f8                                     	mov    rbx,QWORD PTR [rbp-0x8]
   140001b8e:	b9 02 00 00 00                                  	mov    ecx,0x2
   140001b93:	e8 10 16 00 00                                  	call   1400031a8 <__acrt_iob_func>
   140001b98:	48 89 c1                                        	mov    rcx,rax
   140001b9b:	48 8b 45 20                                     	mov    rax,QWORD PTR [rbp+0x20]
   140001b9f:	49 89 d8                                        	mov    r8,rbx
   140001ba2:	48 89 c2                                        	mov    rdx,rax
   140001ba5:	e8 f6 12 00 00                                  	call   140002ea0 <vfprintf>
   140001baa:	e8 c9 16 00 00                                  	call   140003278 <abort>
   140001baf:	90                                              	nop

0000000140001bb0 <mark_section_writable>:
   140001bb0:	55                                              	push   rbp
   140001bb1:	48 89 e5                                        	mov    rbp,rsp
   140001bb4:	48 83 ec 60                                     	sub    rsp,0x60
   140001bb8:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140001bbc:	c7 45 fc 00 00 00 00                            	mov    DWORD PTR [rbp-0x4],0x0
   140001bc3:	e9 82 00 00 00                                  	jmp    140001c4a <mark_section_writable+0x9a>
   140001bc8:	48 8b 0d 21 75 00 00                            	mov    rcx,QWORD PTR [rip+0x7521]        # 1400090f0 <the_secs>
   140001bcf:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001bd2:	48 63 d0                                        	movsxd rdx,eax
   140001bd5:	48 89 d0                                        	mov    rax,rdx
   140001bd8:	48 c1 e0 02                                     	shl    rax,0x2
   140001bdc:	48 01 d0                                        	add    rax,rdx
   140001bdf:	48 c1 e0 03                                     	shl    rax,0x3
   140001be3:	48 01 c8                                        	add    rax,rcx
   140001be6:	48 8b 40 18                                     	mov    rax,QWORD PTR [rax+0x18]
   140001bea:	48 39 45 10                                     	cmp    QWORD PTR [rbp+0x10],rax
   140001bee:	72 56                                           	jb     140001c46 <mark_section_writable+0x96>
   140001bf0:	48 8b 0d f9 74 00 00                            	mov    rcx,QWORD PTR [rip+0x74f9]        # 1400090f0 <the_secs>
   140001bf7:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001bfa:	48 63 d0                                        	movsxd rdx,eax
   140001bfd:	48 89 d0                                        	mov    rax,rdx
   140001c00:	48 c1 e0 02                                     	shl    rax,0x2
   140001c04:	48 01 d0                                        	add    rax,rdx
   140001c07:	48 c1 e0 03                                     	shl    rax,0x3
   140001c0b:	48 01 c8                                        	add    rax,rcx
   140001c0e:	48 8b 48 18                                     	mov    rcx,QWORD PTR [rax+0x18]
   140001c12:	4c 8b 05 d7 74 00 00                            	mov    r8,QWORD PTR [rip+0x74d7]        # 1400090f0 <the_secs>
   140001c19:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001c1c:	48 63 d0                                        	movsxd rdx,eax
   140001c1f:	48 89 d0                                        	mov    rax,rdx
   140001c22:	48 c1 e0 02                                     	shl    rax,0x2
   140001c26:	48 01 d0                                        	add    rax,rdx
   140001c29:	48 c1 e0 03                                     	shl    rax,0x3
   140001c2d:	4c 01 c0                                        	add    rax,r8
   140001c30:	48 8b 40 20                                     	mov    rax,QWORD PTR [rax+0x20]
   140001c34:	8b 40 08                                        	mov    eax,DWORD PTR [rax+0x8]
   140001c37:	89 c0                                           	mov    eax,eax
   140001c39:	48 01 c8                                        	add    rax,rcx
   140001c3c:	48 39 45 10                                     	cmp    QWORD PTR [rbp+0x10],rax
   140001c40:	0f 82 42 02 00 00                               	jb     140001e88 <mark_section_writable+0x2d8>
   140001c46:	83 45 fc 01                                     	add    DWORD PTR [rbp-0x4],0x1
   140001c4a:	8b 05 a8 74 00 00                               	mov    eax,DWORD PTR [rip+0x74a8]        # 1400090f8 <maxSections>
   140001c50:	39 45 fc                                        	cmp    DWORD PTR [rbp-0x4],eax
   140001c53:	0f 8c 6f ff ff ff                               	jl     140001bc8 <mark_section_writable+0x18>
   140001c59:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140001c5d:	48 89 c1                                        	mov    rcx,rax
   140001c60:	e8 26 0f 00 00                                  	call   140002b8b <__mingw_GetSectionForAddress>
   140001c65:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   140001c69:	48 83 7d f0 00                                  	cmp    QWORD PTR [rbp-0x10],0x0
   140001c6e:	75 16                                           	jne    140001c86 <mark_section_writable+0xd6>
   140001c70:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140001c74:	48 89 c2                                        	mov    rdx,rax
   140001c77:	48 8d 05 82 35 00 00                            	lea    rax,[rip+0x3582]        # 140005200 <.rdata+0x20>
   140001c7e:	48 89 c1                                        	mov    rcx,rax
   140001c81:	e8 ba fe ff ff                                  	call   140001b40 <__report_error>
   140001c86:	48 8b 0d 63 74 00 00                            	mov    rcx,QWORD PTR [rip+0x7463]        # 1400090f0 <the_secs>
   140001c8d:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001c90:	48 63 d0                                        	movsxd rdx,eax
   140001c93:	48 89 d0                                        	mov    rax,rdx
   140001c96:	48 c1 e0 02                                     	shl    rax,0x2
   140001c9a:	48 01 d0                                        	add    rax,rdx
   140001c9d:	48 c1 e0 03                                     	shl    rax,0x3
   140001ca1:	48 8d 14 01                                     	lea    rdx,[rcx+rax*1]
   140001ca5:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140001ca9:	48 89 42 20                                     	mov    QWORD PTR [rdx+0x20],rax
   140001cad:	48 8b 0d 3c 74 00 00                            	mov    rcx,QWORD PTR [rip+0x743c]        # 1400090f0 <the_secs>
   140001cb4:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001cb7:	48 63 d0                                        	movsxd rdx,eax
   140001cba:	48 89 d0                                        	mov    rax,rdx
   140001cbd:	48 c1 e0 02                                     	shl    rax,0x2
   140001cc1:	48 01 d0                                        	add    rax,rdx
   140001cc4:	48 c1 e0 03                                     	shl    rax,0x3
   140001cc8:	48 01 c8                                        	add    rax,rcx
   140001ccb:	c7 00 00 00 00 00                               	mov    DWORD PTR [rax],0x0
   140001cd1:	e8 01 10 00 00                                  	call   140002cd7 <_GetPEImageBase>
   140001cd6:	48 89 c1                                        	mov    rcx,rax
   140001cd9:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140001cdd:	8b 40 0c                                        	mov    eax,DWORD PTR [rax+0xc]
   140001ce0:	41 89 c1                                        	mov    r9d,eax
   140001ce3:	4c 8b 05 06 74 00 00                            	mov    r8,QWORD PTR [rip+0x7406]        # 1400090f0 <the_secs>
   140001cea:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001ced:	48 63 d0                                        	movsxd rdx,eax
   140001cf0:	48 89 d0                                        	mov    rax,rdx
   140001cf3:	48 c1 e0 02                                     	shl    rax,0x2
   140001cf7:	48 01 d0                                        	add    rax,rdx
   140001cfa:	48 c1 e0 03                                     	shl    rax,0x3
   140001cfe:	4c 01 c0                                        	add    rax,r8
   140001d01:	4a 8d 14 09                                     	lea    rdx,[rcx+r9*1]
   140001d05:	48 89 50 18                                     	mov    QWORD PTR [rax+0x18],rdx
   140001d09:	48 8b 0d e0 73 00 00                            	mov    rcx,QWORD PTR [rip+0x73e0]        # 1400090f0 <the_secs>
   140001d10:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001d13:	48 63 d0                                        	movsxd rdx,eax
   140001d16:	48 89 d0                                        	mov    rax,rdx
   140001d19:	48 c1 e0 02                                     	shl    rax,0x2
   140001d1d:	48 01 d0                                        	add    rax,rdx
   140001d20:	48 c1 e0 03                                     	shl    rax,0x3
   140001d24:	48 01 c8                                        	add    rax,rcx
   140001d27:	48 8b 40 18                                     	mov    rax,QWORD PTR [rax+0x18]
   140001d2b:	48 8d 55 c0                                     	lea    rdx,[rbp-0x40]
   140001d2f:	41 b8 30 00 00 00                               	mov    r8d,0x30
   140001d35:	48 89 c1                                        	mov    rcx,rax
   140001d38:	48 8b 05 31 85 00 00                            	mov    rax,QWORD PTR [rip+0x8531]        # 14000a270 <__imp_VirtualQuery>
   140001d3f:	ff d0                                           	call   rax
   140001d41:	48 85 c0                                        	test   rax,rax
   140001d44:	75 3d                                           	jne    140001d83 <mark_section_writable+0x1d3>
   140001d46:	48 8b 0d a3 73 00 00                            	mov    rcx,QWORD PTR [rip+0x73a3]        # 1400090f0 <the_secs>
   140001d4d:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001d50:	48 63 d0                                        	movsxd rdx,eax
   140001d53:	48 89 d0                                        	mov    rax,rdx
   140001d56:	48 c1 e0 02                                     	shl    rax,0x2
   140001d5a:	48 01 d0                                        	add    rax,rdx
   140001d5d:	48 c1 e0 03                                     	shl    rax,0x3
   140001d61:	48 01 c8                                        	add    rax,rcx
   140001d64:	48 8b 50 18                                     	mov    rdx,QWORD PTR [rax+0x18]
   140001d68:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140001d6c:	8b 40 08                                        	mov    eax,DWORD PTR [rax+0x8]
   140001d6f:	49 89 d0                                        	mov    r8,rdx
   140001d72:	89 c2                                           	mov    edx,eax
   140001d74:	48 8d 05 a5 34 00 00                            	lea    rax,[rip+0x34a5]        # 140005220 <.rdata+0x40>
   140001d7b:	48 89 c1                                        	mov    rcx,rax
   140001d7e:	e8 bd fd ff ff                                  	call   140001b40 <__report_error>
   140001d83:	8b 45 e4                                        	mov    eax,DWORD PTR [rbp-0x1c]
   140001d86:	83 f8 40                                        	cmp    eax,0x40
   140001d89:	0f 84 e8 00 00 00                               	je     140001e77 <mark_section_writable+0x2c7>
   140001d8f:	8b 45 e4                                        	mov    eax,DWORD PTR [rbp-0x1c]
   140001d92:	83 f8 04                                        	cmp    eax,0x4
   140001d95:	0f 84 dc 00 00 00                               	je     140001e77 <mark_section_writable+0x2c7>
   140001d9b:	8b 45 e4                                        	mov    eax,DWORD PTR [rbp-0x1c]
   140001d9e:	3d 80 00 00 00                                  	cmp    eax,0x80
   140001da3:	0f 84 ce 00 00 00                               	je     140001e77 <mark_section_writable+0x2c7>
   140001da9:	8b 45 e4                                        	mov    eax,DWORD PTR [rbp-0x1c]
   140001dac:	83 f8 08                                        	cmp    eax,0x8
   140001daf:	0f 84 c2 00 00 00                               	je     140001e77 <mark_section_writable+0x2c7>
   140001db5:	8b 45 e4                                        	mov    eax,DWORD PTR [rbp-0x1c]
   140001db8:	83 f8 02                                        	cmp    eax,0x2
   140001dbb:	75 09                                           	jne    140001dc6 <mark_section_writable+0x216>
   140001dbd:	c7 45 f8 04 00 00 00                            	mov    DWORD PTR [rbp-0x8],0x4
   140001dc4:	eb 07                                           	jmp    140001dcd <mark_section_writable+0x21d>
   140001dc6:	c7 45 f8 40 00 00 00                            	mov    DWORD PTR [rbp-0x8],0x40
   140001dcd:	48 8b 0d 1c 73 00 00                            	mov    rcx,QWORD PTR [rip+0x731c]        # 1400090f0 <the_secs>
   140001dd4:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001dd7:	48 63 d0                                        	movsxd rdx,eax
   140001dda:	48 89 d0                                        	mov    rax,rdx
   140001ddd:	48 c1 e0 02                                     	shl    rax,0x2
   140001de1:	48 01 d0                                        	add    rax,rdx
   140001de4:	48 c1 e0 03                                     	shl    rax,0x3
   140001de8:	48 8d 14 01                                     	lea    rdx,[rcx+rax*1]
   140001dec:	48 8b 45 c0                                     	mov    rax,QWORD PTR [rbp-0x40]
   140001df0:	48 89 42 08                                     	mov    QWORD PTR [rdx+0x8],rax
   140001df4:	48 8b 0d f5 72 00 00                            	mov    rcx,QWORD PTR [rip+0x72f5]        # 1400090f0 <the_secs>
   140001dfb:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001dfe:	48 63 d0                                        	movsxd rdx,eax
   140001e01:	48 89 d0                                        	mov    rax,rdx
   140001e04:	48 c1 e0 02                                     	shl    rax,0x2
   140001e08:	48 01 d0                                        	add    rax,rdx
   140001e0b:	48 c1 e0 03                                     	shl    rax,0x3
   140001e0f:	48 8d 14 01                                     	lea    rdx,[rcx+rax*1]
   140001e13:	48 8b 45 d8                                     	mov    rax,QWORD PTR [rbp-0x28]
   140001e17:	48 89 42 10                                     	mov    QWORD PTR [rdx+0x10],rax
   140001e1b:	48 8b 0d ce 72 00 00                            	mov    rcx,QWORD PTR [rip+0x72ce]        # 1400090f0 <the_secs>
   140001e22:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001e25:	48 63 d0                                        	movsxd rdx,eax
   140001e28:	48 89 d0                                        	mov    rax,rdx
   140001e2b:	48 c1 e0 02                                     	shl    rax,0x2
   140001e2f:	48 01 d0                                        	add    rax,rdx
   140001e32:	48 c1 e0 03                                     	shl    rax,0x3
   140001e36:	48 01 c8                                        	add    rax,rcx
   140001e39:	49 89 c0                                        	mov    r8,rax
   140001e3c:	48 8b 55 d8                                     	mov    rdx,QWORD PTR [rbp-0x28]
   140001e40:	48 8b 45 c0                                     	mov    rax,QWORD PTR [rbp-0x40]
   140001e44:	8b 4d f8                                        	mov    ecx,DWORD PTR [rbp-0x8]
   140001e47:	4d 89 c1                                        	mov    r9,r8
   140001e4a:	41 89 c8                                        	mov    r8d,ecx
   140001e4d:	48 89 c1                                        	mov    rcx,rax
   140001e50:	48 8b 05 11 84 00 00                            	mov    rax,QWORD PTR [rip+0x8411]        # 14000a268 <__imp_VirtualProtect>
   140001e57:	ff d0                                           	call   rax
   140001e59:	85 c0                                           	test   eax,eax
   140001e5b:	75 1a                                           	jne    140001e77 <mark_section_writable+0x2c7>
   140001e5d:	48 8b 05 b4 83 00 00                            	mov    rax,QWORD PTR [rip+0x83b4]        # 14000a218 <__imp_GetLastError>
   140001e64:	ff d0                                           	call   rax
   140001e66:	89 c2                                           	mov    edx,eax
   140001e68:	48 8d 05 e9 33 00 00                            	lea    rax,[rip+0x33e9]        # 140005258 <.rdata+0x78>
   140001e6f:	48 89 c1                                        	mov    rcx,rax
   140001e72:	e8 c9 fc ff ff                                  	call   140001b40 <__report_error>
   140001e77:	8b 05 7b 72 00 00                               	mov    eax,DWORD PTR [rip+0x727b]        # 1400090f8 <maxSections>
   140001e7d:	83 c0 01                                        	add    eax,0x1
   140001e80:	89 05 72 72 00 00                               	mov    DWORD PTR [rip+0x7272],eax        # 1400090f8 <maxSections>
   140001e86:	eb 01                                           	jmp    140001e89 <mark_section_writable+0x2d9>
   140001e88:	90                                              	nop
   140001e89:	48 83 c4 60                                     	add    rsp,0x60
   140001e8d:	5d                                              	pop    rbp
   140001e8e:	c3                                              	ret

0000000140001e8f <restore_modified_sections>:
   140001e8f:	55                                              	push   rbp
   140001e90:	48 89 e5                                        	mov    rbp,rsp
   140001e93:	48 83 ec 30                                     	sub    rsp,0x30
   140001e97:	c7 45 fc 00 00 00 00                            	mov    DWORD PTR [rbp-0x4],0x0
   140001e9e:	e9 ad 00 00 00                                  	jmp    140001f50 <restore_modified_sections+0xc1>
   140001ea3:	48 8b 0d 46 72 00 00                            	mov    rcx,QWORD PTR [rip+0x7246]        # 1400090f0 <the_secs>
   140001eaa:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001ead:	48 63 d0                                        	movsxd rdx,eax
   140001eb0:	48 89 d0                                        	mov    rax,rdx
   140001eb3:	48 c1 e0 02                                     	shl    rax,0x2
   140001eb7:	48 01 d0                                        	add    rax,rdx
   140001eba:	48 c1 e0 03                                     	shl    rax,0x3
   140001ebe:	48 01 c8                                        	add    rax,rcx
   140001ec1:	8b 00                                           	mov    eax,DWORD PTR [rax]
   140001ec3:	85 c0                                           	test   eax,eax
   140001ec5:	0f 84 80 00 00 00                               	je     140001f4b <restore_modified_sections+0xbc>
   140001ecb:	48 8b 0d 1e 72 00 00                            	mov    rcx,QWORD PTR [rip+0x721e]        # 1400090f0 <the_secs>
   140001ed2:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001ed5:	48 63 d0                                        	movsxd rdx,eax
   140001ed8:	48 89 d0                                        	mov    rax,rdx
   140001edb:	48 c1 e0 02                                     	shl    rax,0x2
   140001edf:	48 01 d0                                        	add    rax,rdx
   140001ee2:	48 c1 e0 03                                     	shl    rax,0x3
   140001ee6:	48 01 c8                                        	add    rax,rcx
   140001ee9:	44 8b 10                                        	mov    r10d,DWORD PTR [rax]
   140001eec:	48 8b 0d fd 71 00 00                            	mov    rcx,QWORD PTR [rip+0x71fd]        # 1400090f0 <the_secs>
   140001ef3:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001ef6:	48 63 d0                                        	movsxd rdx,eax
   140001ef9:	48 89 d0                                        	mov    rax,rdx
   140001efc:	48 c1 e0 02                                     	shl    rax,0x2
   140001f00:	48 01 d0                                        	add    rax,rdx
   140001f03:	48 c1 e0 03                                     	shl    rax,0x3
   140001f07:	48 01 c8                                        	add    rax,rcx
   140001f0a:	48 8b 48 10                                     	mov    rcx,QWORD PTR [rax+0x10]
   140001f0e:	4c 8b 05 db 71 00 00                            	mov    r8,QWORD PTR [rip+0x71db]        # 1400090f0 <the_secs>
   140001f15:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140001f18:	48 63 d0                                        	movsxd rdx,eax
   140001f1b:	48 89 d0                                        	mov    rax,rdx
   140001f1e:	48 c1 e0 02                                     	shl    rax,0x2
   140001f22:	48 01 d0                                        	add    rax,rdx
   140001f25:	48 c1 e0 03                                     	shl    rax,0x3
   140001f29:	4c 01 c0                                        	add    rax,r8
   140001f2c:	48 8b 40 08                                     	mov    rax,QWORD PTR [rax+0x8]
   140001f30:	48 8d 55 f8                                     	lea    rdx,[rbp-0x8]
   140001f34:	49 89 d1                                        	mov    r9,rdx
   140001f37:	45 89 d0                                        	mov    r8d,r10d
   140001f3a:	48 89 ca                                        	mov    rdx,rcx
   140001f3d:	48 89 c1                                        	mov    rcx,rax
   140001f40:	48 8b 05 21 83 00 00                            	mov    rax,QWORD PTR [rip+0x8321]        # 14000a268 <__imp_VirtualProtect>
   140001f47:	ff d0                                           	call   rax
   140001f49:	eb 01                                           	jmp    140001f4c <restore_modified_sections+0xbd>
   140001f4b:	90                                              	nop
   140001f4c:	83 45 fc 01                                     	add    DWORD PTR [rbp-0x4],0x1
   140001f50:	8b 05 a2 71 00 00                               	mov    eax,DWORD PTR [rip+0x71a2]        # 1400090f8 <maxSections>
   140001f56:	39 45 fc                                        	cmp    DWORD PTR [rbp-0x4],eax
   140001f59:	0f 8c 44 ff ff ff                               	jl     140001ea3 <restore_modified_sections+0x14>
   140001f5f:	90                                              	nop
   140001f60:	90                                              	nop
   140001f61:	48 83 c4 30                                     	add    rsp,0x30
   140001f65:	5d                                              	pop    rbp
   140001f66:	c3                                              	ret

0000000140001f67 <__write_memory>:
   140001f67:	55                                              	push   rbp
   140001f68:	48 89 e5                                        	mov    rbp,rsp
   140001f6b:	48 83 ec 20                                     	sub    rsp,0x20
   140001f6f:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140001f73:	48 89 55 18                                     	mov    QWORD PTR [rbp+0x18],rdx
   140001f77:	4c 89 45 20                                     	mov    QWORD PTR [rbp+0x20],r8
   140001f7b:	48 83 7d 20 00                                  	cmp    QWORD PTR [rbp+0x20],0x0
   140001f80:	74 25                                           	je     140001fa7 <__write_memory+0x40>
   140001f82:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140001f86:	48 89 c1                                        	mov    rcx,rax
   140001f89:	e8 22 fc ff ff                                  	call   140001bb0 <mark_section_writable>
   140001f8e:	48 8b 4d 20                                     	mov    rcx,QWORD PTR [rbp+0x20]
   140001f92:	48 8b 55 18                                     	mov    rdx,QWORD PTR [rbp+0x18]
   140001f96:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140001f9a:	49 89 c8                                        	mov    r8,rcx
   140001f9d:	48 89 c1                                        	mov    rcx,rax
   140001fa0:	e8 03 13 00 00                                  	call   1400032a8 <memcpy>
   140001fa5:	eb 01                                           	jmp    140001fa8 <__write_memory+0x41>
   140001fa7:	90                                              	nop
   140001fa8:	48 83 c4 20                                     	add    rsp,0x20
   140001fac:	5d                                              	pop    rbp
   140001fad:	c3                                              	ret

0000000140001fae <do_pseudo_reloc>:
   140001fae:	55                                              	push   rbp
   140001faf:	48 89 e5                                        	mov    rbp,rsp
   140001fb2:	48 83 c4 80                                     	add    rsp,0xffffffffffffff80
   140001fb6:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140001fba:	48 89 55 18                                     	mov    QWORD PTR [rbp+0x18],rdx
   140001fbe:	4c 89 45 20                                     	mov    QWORD PTR [rbp+0x20],r8
   140001fc2:	48 8b 45 18                                     	mov    rax,QWORD PTR [rbp+0x18]
   140001fc6:	48 2b 45 10                                     	sub    rax,QWORD PTR [rbp+0x10]
   140001fca:	48 89 45 e0                                     	mov    QWORD PTR [rbp-0x20],rax
   140001fce:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140001fd2:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140001fd6:	48 83 7d e0 07                                  	cmp    QWORD PTR [rbp-0x20],0x7
   140001fdb:	0f 8e 50 03 00 00                               	jle    140002331 <do_pseudo_reloc+0x383>
   140001fe1:	48 83 7d e0 0b                                  	cmp    QWORD PTR [rbp-0x20],0xb
   140001fe6:	7e 25                                           	jle    14000200d <do_pseudo_reloc+0x5f>
   140001fe8:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140001fec:	8b 00                                           	mov    eax,DWORD PTR [rax]
   140001fee:	85 c0                                           	test   eax,eax
   140001ff0:	75 1b                                           	jne    14000200d <do_pseudo_reloc+0x5f>
   140001ff2:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140001ff6:	8b 40 04                                        	mov    eax,DWORD PTR [rax+0x4]
   140001ff9:	85 c0                                           	test   eax,eax
   140001ffb:	75 10                                           	jne    14000200d <do_pseudo_reloc+0x5f>
   140001ffd:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002001:	8b 40 08                                        	mov    eax,DWORD PTR [rax+0x8]
   140002004:	85 c0                                           	test   eax,eax
   140002006:	75 05                                           	jne    14000200d <do_pseudo_reloc+0x5f>
   140002008:	48 83 45 f8 0c                                  	add    QWORD PTR [rbp-0x8],0xc
   14000200d:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002011:	8b 00                                           	mov    eax,DWORD PTR [rax]
   140002013:	85 c0                                           	test   eax,eax
   140002015:	75 0b                                           	jne    140002022 <do_pseudo_reloc+0x74>
   140002017:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   14000201b:	8b 40 04                                        	mov    eax,DWORD PTR [rax+0x4]
   14000201e:	85 c0                                           	test   eax,eax
   140002020:	74 59                                           	je     14000207b <do_pseudo_reloc+0xcd>
   140002022:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002026:	48 89 45 e8                                     	mov    QWORD PTR [rbp-0x18],rax
   14000202a:	eb 40                                           	jmp    14000206c <do_pseudo_reloc+0xbe>
   14000202c:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002030:	8b 40 04                                        	mov    eax,DWORD PTR [rax+0x4]
   140002033:	89 c2                                           	mov    edx,eax
   140002035:	48 8b 45 20                                     	mov    rax,QWORD PTR [rbp+0x20]
   140002039:	48 01 d0                                        	add    rax,rdx
   14000203c:	48 89 45 e0                                     	mov    QWORD PTR [rbp-0x20],rax
   140002040:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140002044:	8b 10                                           	mov    edx,DWORD PTR [rax]
   140002046:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   14000204a:	8b 00                                           	mov    eax,DWORD PTR [rax]
   14000204c:	01 d0                                           	add    eax,edx
   14000204e:	89 45 b4                                        	mov    DWORD PTR [rbp-0x4c],eax
   140002051:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140002055:	48 8d 55 b4                                     	lea    rdx,[rbp-0x4c]
   140002059:	41 b8 04 00 00 00                               	mov    r8d,0x4
   14000205f:	48 89 c1                                        	mov    rcx,rax
   140002062:	e8 00 ff ff ff                                  	call   140001f67 <__write_memory>
   140002067:	48 83 45 e8 08                                  	add    QWORD PTR [rbp-0x18],0x8
   14000206c:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002070:	48 3b 45 18                                     	cmp    rax,QWORD PTR [rbp+0x18]
   140002074:	72 b6                                           	jb     14000202c <do_pseudo_reloc+0x7e>
   140002076:	e9 b7 02 00 00                                  	jmp    140002332 <do_pseudo_reloc+0x384>
   14000207b:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   14000207f:	8b 40 08                                        	mov    eax,DWORD PTR [rax+0x8]
   140002082:	83 f8 01                                        	cmp    eax,0x1
   140002085:	74 18                                           	je     14000209f <do_pseudo_reloc+0xf1>
   140002087:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   14000208b:	8b 40 08                                        	mov    eax,DWORD PTR [rax+0x8]
   14000208e:	89 c2                                           	mov    edx,eax
   140002090:	48 8d 05 e9 31 00 00                            	lea    rax,[rip+0x31e9]        # 140005280 <.rdata+0xa0>
   140002097:	48 89 c1                                        	mov    rcx,rax
   14000209a:	e8 a1 fa ff ff                                  	call   140001b40 <__report_error>
   14000209f:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   1400020a3:	48 83 c0 0c                                     	add    rax,0xc
   1400020a7:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   1400020ab:	e9 71 02 00 00                                  	jmp    140002321 <do_pseudo_reloc+0x373>
   1400020b0:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400020b4:	8b 40 04                                        	mov    eax,DWORD PTR [rax+0x4]
   1400020b7:	89 c2                                           	mov    edx,eax
   1400020b9:	48 8b 45 20                                     	mov    rax,QWORD PTR [rbp+0x20]
   1400020bd:	48 01 d0                                        	add    rax,rdx
   1400020c0:	48 89 45 e0                                     	mov    QWORD PTR [rbp-0x20],rax
   1400020c4:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400020c8:	8b 00                                           	mov    eax,DWORD PTR [rax]
   1400020ca:	89 c2                                           	mov    edx,eax
   1400020cc:	48 8b 45 20                                     	mov    rax,QWORD PTR [rbp+0x20]
   1400020d0:	48 01 d0                                        	add    rax,rdx
   1400020d3:	48 89 45 d8                                     	mov    QWORD PTR [rbp-0x28],rax
   1400020d7:	48 8b 45 d8                                     	mov    rax,QWORD PTR [rbp-0x28]
   1400020db:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   1400020de:	48 89 45 d8                                     	mov    QWORD PTR [rbp-0x28],rax
   1400020e2:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400020e6:	8b 40 08                                        	mov    eax,DWORD PTR [rax+0x8]
   1400020e9:	0f b6 c0                                        	movzx  eax,al
   1400020ec:	83 f8 40                                        	cmp    eax,0x40
   1400020ef:	0f 84 b6 00 00 00                               	je     1400021ab <do_pseudo_reloc+0x1fd>
   1400020f5:	83 f8 40                                        	cmp    eax,0x40
   1400020f8:	0f 87 ba 00 00 00                               	ja     1400021b8 <do_pseudo_reloc+0x20a>
   1400020fe:	83 f8 20                                        	cmp    eax,0x20
   140002101:	74 77                                           	je     14000217a <do_pseudo_reloc+0x1cc>
   140002103:	83 f8 20                                        	cmp    eax,0x20
   140002106:	0f 87 ac 00 00 00                               	ja     1400021b8 <do_pseudo_reloc+0x20a>
   14000210c:	83 f8 08                                        	cmp    eax,0x8
   14000210f:	74 0a                                           	je     14000211b <do_pseudo_reloc+0x16d>
   140002111:	83 f8 10                                        	cmp    eax,0x10
   140002114:	74 38                                           	je     14000214e <do_pseudo_reloc+0x1a0>
   140002116:	e9 9d 00 00 00                                  	jmp    1400021b8 <do_pseudo_reloc+0x20a>
   14000211b:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   14000211f:	0f b6 00                                        	movzx  eax,BYTE PTR [rax]
   140002122:	0f b6 c0                                        	movzx  eax,al
   140002125:	48 89 45 b8                                     	mov    QWORD PTR [rbp-0x48],rax
   140002129:	48 8b 45 b8                                     	mov    rax,QWORD PTR [rbp-0x48]
   14000212d:	25 80 00 00 00                                  	and    eax,0x80
   140002132:	48 85 c0                                        	test   rax,rax
   140002135:	0f 84 a0 00 00 00                               	je     1400021db <do_pseudo_reloc+0x22d>
   14000213b:	48 8b 45 b8                                     	mov    rax,QWORD PTR [rbp-0x48]
   14000213f:	48 0d 00 ff ff ff                               	or     rax,0xffffffffffffff00
   140002145:	48 89 45 b8                                     	mov    QWORD PTR [rbp-0x48],rax
   140002149:	e9 8d 00 00 00                                  	jmp    1400021db <do_pseudo_reloc+0x22d>
   14000214e:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140002152:	0f b7 00                                        	movzx  eax,WORD PTR [rax]
   140002155:	0f b7 c0                                        	movzx  eax,ax
   140002158:	48 89 45 b8                                     	mov    QWORD PTR [rbp-0x48],rax
   14000215c:	48 8b 45 b8                                     	mov    rax,QWORD PTR [rbp-0x48]
   140002160:	25 00 80 00 00                                  	and    eax,0x8000
   140002165:	48 85 c0                                        	test   rax,rax
   140002168:	74 74                                           	je     1400021de <do_pseudo_reloc+0x230>
   14000216a:	48 8b 45 b8                                     	mov    rax,QWORD PTR [rbp-0x48]
   14000216e:	48 0d 00 00 ff ff                               	or     rax,0xffffffffffff0000
   140002174:	48 89 45 b8                                     	mov    QWORD PTR [rbp-0x48],rax
   140002178:	eb 64                                           	jmp    1400021de <do_pseudo_reloc+0x230>
   14000217a:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   14000217e:	8b 00                                           	mov    eax,DWORD PTR [rax]
   140002180:	89 c0                                           	mov    eax,eax
   140002182:	48 89 45 b8                                     	mov    QWORD PTR [rbp-0x48],rax
   140002186:	48 8b 45 b8                                     	mov    rax,QWORD PTR [rbp-0x48]
   14000218a:	25 00 00 00 80                                  	and    eax,0x80000000
   14000218f:	48 85 c0                                        	test   rax,rax
   140002192:	74 4d                                           	je     1400021e1 <do_pseudo_reloc+0x233>
   140002194:	48 8b 45 b8                                     	mov    rax,QWORD PTR [rbp-0x48]
   140002198:	48 ba 00 00 00 00 ff ff ff ff                   	movabs rdx,0xffffffff00000000
   1400021a2:	48 09 d0                                        	or     rax,rdx
   1400021a5:	48 89 45 b8                                     	mov    QWORD PTR [rbp-0x48],rax
   1400021a9:	eb 36                                           	jmp    1400021e1 <do_pseudo_reloc+0x233>
   1400021ab:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   1400021af:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   1400021b2:	48 89 45 b8                                     	mov    QWORD PTR [rbp-0x48],rax
   1400021b6:	eb 2a                                           	jmp    1400021e2 <do_pseudo_reloc+0x234>
   1400021b8:	48 c7 45 b8 00 00 00 00                         	mov    QWORD PTR [rbp-0x48],0x0
   1400021c0:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400021c4:	8b 40 08                                        	mov    eax,DWORD PTR [rax+0x8]
   1400021c7:	0f b6 c0                                        	movzx  eax,al
   1400021ca:	89 c2                                           	mov    edx,eax
   1400021cc:	48 8d 05 e5 30 00 00                            	lea    rax,[rip+0x30e5]        # 1400052b8 <.rdata+0xd8>
   1400021d3:	48 89 c1                                        	mov    rcx,rax
   1400021d6:	e8 65 f9 ff ff                                  	call   140001b40 <__report_error>
   1400021db:	90                                              	nop
   1400021dc:	eb 04                                           	jmp    1400021e2 <do_pseudo_reloc+0x234>
   1400021de:	90                                              	nop
   1400021df:	eb 01                                           	jmp    1400021e2 <do_pseudo_reloc+0x234>
   1400021e1:	90                                              	nop
   1400021e2:	48 8b 4d b8                                     	mov    rcx,QWORD PTR [rbp-0x48]
   1400021e6:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400021ea:	8b 00                                           	mov    eax,DWORD PTR [rax]
   1400021ec:	89 c2                                           	mov    edx,eax
   1400021ee:	48 8b 45 20                                     	mov    rax,QWORD PTR [rbp+0x20]
   1400021f2:	48 01 c2                                        	add    rdx,rax
   1400021f5:	48 89 c8                                        	mov    rax,rcx
   1400021f8:	48 29 d0                                        	sub    rax,rdx
   1400021fb:	48 89 45 b8                                     	mov    QWORD PTR [rbp-0x48],rax
   1400021ff:	48 8b 55 b8                                     	mov    rdx,QWORD PTR [rbp-0x48]
   140002203:	48 8b 45 d8                                     	mov    rax,QWORD PTR [rbp-0x28]
   140002207:	48 01 d0                                        	add    rax,rdx
   14000220a:	48 89 45 b8                                     	mov    QWORD PTR [rbp-0x48],rax
   14000220e:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140002212:	8b 40 08                                        	mov    eax,DWORD PTR [rax+0x8]
   140002215:	25 ff 00 00 00                                  	and    eax,0xff
   14000221a:	89 45 d4                                        	mov    DWORD PTR [rbp-0x2c],eax
   14000221d:	83 7d d4 3f                                     	cmp    DWORD PTR [rbp-0x2c],0x3f
   140002221:	77 70                                           	ja     140002293 <do_pseudo_reloc+0x2e5>
   140002223:	8b 45 d4                                        	mov    eax,DWORD PTR [rbp-0x2c]
   140002226:	ba 01 00 00 00                                  	mov    edx,0x1
   14000222b:	89 c1                                           	mov    ecx,eax
   14000222d:	48 d3 e2                                        	shl    rdx,cl
   140002230:	48 89 d0                                        	mov    rax,rdx
   140002233:	48 83 e8 01                                     	sub    rax,0x1
   140002237:	48 89 45 c8                                     	mov    QWORD PTR [rbp-0x38],rax
   14000223b:	8b 45 d4                                        	mov    eax,DWORD PTR [rbp-0x2c]
   14000223e:	83 e8 01                                        	sub    eax,0x1
   140002241:	48 c7 c2 ff ff ff ff                            	mov    rdx,0xffffffffffffffff
   140002248:	89 c1                                           	mov    ecx,eax
   14000224a:	48 d3 e2                                        	shl    rdx,cl
   14000224d:	48 89 d0                                        	mov    rax,rdx
   140002250:	48 89 45 c0                                     	mov    QWORD PTR [rbp-0x40],rax
   140002254:	48 8b 45 b8                                     	mov    rax,QWORD PTR [rbp-0x48]
   140002258:	48 39 45 c8                                     	cmp    QWORD PTR [rbp-0x38],rax
   14000225c:	7c 0a                                           	jl     140002268 <do_pseudo_reloc+0x2ba>
   14000225e:	48 8b 45 b8                                     	mov    rax,QWORD PTR [rbp-0x48]
   140002262:	48 39 45 c0                                     	cmp    QWORD PTR [rbp-0x40],rax
   140002266:	7e 2b                                           	jle    140002293 <do_pseudo_reloc+0x2e5>
   140002268:	48 8b 55 b8                                     	mov    rdx,QWORD PTR [rbp-0x48]
   14000226c:	4c 8b 45 d8                                     	mov    r8,QWORD PTR [rbp-0x28]
   140002270:	48 8b 4d e0                                     	mov    rcx,QWORD PTR [rbp-0x20]
   140002274:	8b 45 d4                                        	mov    eax,DWORD PTR [rbp-0x2c]
   140002277:	48 89 54 24 20                                  	mov    QWORD PTR [rsp+0x20],rdx
   14000227c:	4d 89 c1                                        	mov    r9,r8
   14000227f:	49 89 c8                                        	mov    r8,rcx
   140002282:	89 c2                                           	mov    edx,eax
   140002284:	48 8d 05 5d 30 00 00                            	lea    rax,[rip+0x305d]        # 1400052e8 <.rdata+0x108>
   14000228b:	48 89 c1                                        	mov    rcx,rax
   14000228e:	e8 ad f8 ff ff                                  	call   140001b40 <__report_error>
   140002293:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140002297:	8b 40 08                                        	mov    eax,DWORD PTR [rax+0x8]
   14000229a:	0f b6 c0                                        	movzx  eax,al
   14000229d:	83 f8 40                                        	cmp    eax,0x40
   1400022a0:	74 63                                           	je     140002305 <do_pseudo_reloc+0x357>
   1400022a2:	83 f8 40                                        	cmp    eax,0x40
   1400022a5:	77 75                                           	ja     14000231c <do_pseudo_reloc+0x36e>
   1400022a7:	83 f8 20                                        	cmp    eax,0x20
   1400022aa:	74 41                                           	je     1400022ed <do_pseudo_reloc+0x33f>
   1400022ac:	83 f8 20                                        	cmp    eax,0x20
   1400022af:	77 6b                                           	ja     14000231c <do_pseudo_reloc+0x36e>
   1400022b1:	83 f8 08                                        	cmp    eax,0x8
   1400022b4:	74 07                                           	je     1400022bd <do_pseudo_reloc+0x30f>
   1400022b6:	83 f8 10                                        	cmp    eax,0x10
   1400022b9:	74 1a                                           	je     1400022d5 <do_pseudo_reloc+0x327>
   1400022bb:	eb 5f                                           	jmp    14000231c <do_pseudo_reloc+0x36e>
   1400022bd:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   1400022c1:	48 8d 55 b8                                     	lea    rdx,[rbp-0x48]
   1400022c5:	41 b8 01 00 00 00                               	mov    r8d,0x1
   1400022cb:	48 89 c1                                        	mov    rcx,rax
   1400022ce:	e8 94 fc ff ff                                  	call   140001f67 <__write_memory>
   1400022d3:	eb 47                                           	jmp    14000231c <do_pseudo_reloc+0x36e>
   1400022d5:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   1400022d9:	48 8d 55 b8                                     	lea    rdx,[rbp-0x48]
   1400022dd:	41 b8 02 00 00 00                               	mov    r8d,0x2
   1400022e3:	48 89 c1                                        	mov    rcx,rax
   1400022e6:	e8 7c fc ff ff                                  	call   140001f67 <__write_memory>
   1400022eb:	eb 2f                                           	jmp    14000231c <do_pseudo_reloc+0x36e>
   1400022ed:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   1400022f1:	48 8d 55 b8                                     	lea    rdx,[rbp-0x48]
   1400022f5:	41 b8 04 00 00 00                               	mov    r8d,0x4
   1400022fb:	48 89 c1                                        	mov    rcx,rax
   1400022fe:	e8 64 fc ff ff                                  	call   140001f67 <__write_memory>
   140002303:	eb 17                                           	jmp    14000231c <do_pseudo_reloc+0x36e>
   140002305:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140002309:	48 8d 55 b8                                     	lea    rdx,[rbp-0x48]
   14000230d:	41 b8 08 00 00 00                               	mov    r8d,0x8
   140002313:	48 89 c1                                        	mov    rcx,rax
   140002316:	e8 4c fc ff ff                                  	call   140001f67 <__write_memory>
   14000231b:	90                                              	nop
   14000231c:	48 83 45 f0 0c                                  	add    QWORD PTR [rbp-0x10],0xc
   140002321:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140002325:	48 3b 45 18                                     	cmp    rax,QWORD PTR [rbp+0x18]
   140002329:	0f 82 81 fd ff ff                               	jb     1400020b0 <do_pseudo_reloc+0x102>
   14000232f:	eb 01                                           	jmp    140002332 <do_pseudo_reloc+0x384>
   140002331:	90                                              	nop
   140002332:	48 83 ec 80                                     	sub    rsp,0xffffffffffffff80
   140002336:	5d                                              	pop    rbp
   140002337:	c3                                              	ret

0000000140002338 <_pei386_runtime_relocator>:
   140002338:	55                                              	push   rbp
   140002339:	48 89 e5                                        	mov    rbp,rsp
   14000233c:	48 83 ec 30                                     	sub    rsp,0x30
   140002340:	8b 05 b6 6d 00 00                               	mov    eax,DWORD PTR [rip+0x6db6]        # 1400090fc <was_init.0>
   140002346:	85 c0                                           	test   eax,eax
   140002348:	0f 85 88 00 00 00                               	jne    1400023d6 <_pei386_runtime_relocator+0x9e>
   14000234e:	8b 05 a8 6d 00 00                               	mov    eax,DWORD PTR [rip+0x6da8]        # 1400090fc <was_init.0>
   140002354:	83 c0 01                                        	add    eax,0x1
   140002357:	89 05 9f 6d 00 00                               	mov    DWORD PTR [rip+0x6d9f],eax        # 1400090fc <was_init.0>
   14000235d:	e8 79 08 00 00                                  	call   140002bdb <__mingw_GetSectionCount>
   140002362:	89 45 fc                                        	mov    DWORD PTR [rbp-0x4],eax
   140002365:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140002368:	48 63 d0                                        	movsxd rdx,eax
   14000236b:	48 89 d0                                        	mov    rax,rdx
   14000236e:	48 c1 e0 02                                     	shl    rax,0x2
   140002372:	48 01 d0                                        	add    rax,rdx
   140002375:	48 c1 e0 03                                     	shl    rax,0x3
   140002379:	48 83 c0 0f                                     	add    rax,0xf
   14000237d:	48 c1 e8 04                                     	shr    rax,0x4
   140002381:	48 c1 e0 04                                     	shl    rax,0x4
   140002385:	e8 d6 0a 00 00                                  	call   140002e60 <___chkstk_ms>
   14000238a:	48 29 c4                                        	sub    rsp,rax
   14000238d:	48 8d 44 24 20                                  	lea    rax,[rsp+0x20]
   140002392:	48 83 c0 0f                                     	add    rax,0xf
   140002396:	48 c1 e8 04                                     	shr    rax,0x4
   14000239a:	48 c1 e0 04                                     	shl    rax,0x4
   14000239e:	48 89 05 4b 6d 00 00                            	mov    QWORD PTR [rip+0x6d4b],rax        # 1400090f0 <the_secs>
   1400023a5:	c7 05 49 6d 00 00 00 00 00 00                   	mov    DWORD PTR [rip+0x6d49],0x0        # 1400090f8 <maxSections>
   1400023af:	4c 8b 05 4a 30 00 00                            	mov    r8,QWORD PTR [rip+0x304a]        # 140005400 <.refptr.__image_base__>
   1400023b6:	48 8b 05 03 30 00 00                            	mov    rax,QWORD PTR [rip+0x3003]        # 1400053c0 <.refptr.__RUNTIME_PSEUDO_RELOC_LIST_END__>
   1400023bd:	48 89 c2                                        	mov    rdx,rax
   1400023c0:	48 8b 05 09 30 00 00                            	mov    rax,QWORD PTR [rip+0x3009]        # 1400053d0 <.refptr.__RUNTIME_PSEUDO_RELOC_LIST__>
   1400023c7:	48 89 c1                                        	mov    rcx,rax
   1400023ca:	e8 df fb ff ff                                  	call   140001fae <do_pseudo_reloc>
   1400023cf:	e8 bb fa ff ff                                  	call   140001e8f <restore_modified_sections>
   1400023d4:	eb 01                                           	jmp    1400023d7 <_pei386_runtime_relocator+0x9f>
   1400023d6:	90                                              	nop
   1400023d7:	48 89 ec                                        	mov    rsp,rbp
   1400023da:	5d                                              	pop    rbp
   1400023db:	c3                                              	ret
   1400023dc:	90                                              	nop
   1400023dd:	90                                              	nop
   1400023de:	90                                              	nop
   1400023df:	90                                              	nop

00000001400023e0 <__mingw_raise_matherr>:
   1400023e0:	55                                              	push   rbp
   1400023e1:	48 89 e5                                        	mov    rbp,rsp
   1400023e4:	48 83 ec 50                                     	sub    rsp,0x50
   1400023e8:	89 4d 10                                        	mov    DWORD PTR [rbp+0x10],ecx
   1400023eb:	48 89 55 18                                     	mov    QWORD PTR [rbp+0x18],rdx
   1400023ef:	f2 0f 11 55 20                                  	movsd  QWORD PTR [rbp+0x20],xmm2
   1400023f4:	f2 0f 11 5d 28                                  	movsd  QWORD PTR [rbp+0x28],xmm3
   1400023f9:	48 8b 05 00 6d 00 00                            	mov    rax,QWORD PTR [rip+0x6d00]        # 140009100 <stUserMathErr>
   140002400:	48 85 c0                                        	test   rax,rax
   140002403:	74 3e                                           	je     140002443 <__mingw_raise_matherr+0x63>
   140002405:	8b 45 10                                        	mov    eax,DWORD PTR [rbp+0x10]
   140002408:	89 45 d0                                        	mov    DWORD PTR [rbp-0x30],eax
   14000240b:	48 8b 45 18                                     	mov    rax,QWORD PTR [rbp+0x18]
   14000240f:	48 89 45 d8                                     	mov    QWORD PTR [rbp-0x28],rax
   140002413:	f2 0f 10 45 20                                  	movsd  xmm0,QWORD PTR [rbp+0x20]
   140002418:	f2 0f 11 45 e0                                  	movsd  QWORD PTR [rbp-0x20],xmm0
   14000241d:	f2 0f 10 45 28                                  	movsd  xmm0,QWORD PTR [rbp+0x28]
   140002422:	f2 0f 11 45 e8                                  	movsd  QWORD PTR [rbp-0x18],xmm0
   140002427:	f2 0f 10 45 30                                  	movsd  xmm0,QWORD PTR [rbp+0x30]
   14000242c:	f2 0f 11 45 f0                                  	movsd  QWORD PTR [rbp-0x10],xmm0
   140002431:	48 8b 15 c8 6c 00 00                            	mov    rdx,QWORD PTR [rip+0x6cc8]        # 140009100 <stUserMathErr>
   140002438:	48 8d 45 d0                                     	lea    rax,[rbp-0x30]
   14000243c:	48 89 c1                                        	mov    rcx,rax
   14000243f:	ff d2                                           	call   rdx
   140002441:	eb 01                                           	jmp    140002444 <__mingw_raise_matherr+0x64>
   140002443:	90                                              	nop
   140002444:	48 83 c4 50                                     	add    rsp,0x50
   140002448:	5d                                              	pop    rbp
   140002449:	c3                                              	ret

000000014000244a <__mingw_setusermatherr>:
   14000244a:	55                                              	push   rbp
   14000244b:	48 89 e5                                        	mov    rbp,rsp
   14000244e:	48 83 ec 20                                     	sub    rsp,0x20
   140002452:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140002456:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   14000245a:	48 89 05 9f 6c 00 00                            	mov    QWORD PTR [rip+0x6c9f],rax        # 140009100 <stUserMathErr>
   140002461:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140002465:	48 89 c1                                        	mov    rcx,rax
   140002468:	e8 8b 0d 00 00                                  	call   1400031f8 <__setusermatherr>
   14000246d:	90                                              	nop
   14000246e:	48 83 c4 20                                     	add    rsp,0x20
   140002472:	5d                                              	pop    rbp
   140002473:	c3                                              	ret
   140002474:	90                                              	nop
   140002475:	90                                              	nop
   140002476:	90                                              	nop
   140002477:	90                                              	nop
   140002478:	90                                              	nop
   140002479:	90                                              	nop
   14000247a:	90                                              	nop
   14000247b:	90                                              	nop
   14000247c:	90                                              	nop
   14000247d:	90                                              	nop
   14000247e:	90                                              	nop
   14000247f:	90                                              	nop

0000000140002480 <_gnu_exception_handler>:
   140002480:	55                                              	push   rbp
   140002481:	48 89 e5                                        	mov    rbp,rsp
   140002484:	48 83 ec 30                                     	sub    rsp,0x30
   140002488:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   14000248c:	c7 45 fc 00 00 00 00                            	mov    DWORD PTR [rbp-0x4],0x0
   140002493:	c7 45 f8 00 00 00 00                            	mov    DWORD PTR [rbp-0x8],0x0
   14000249a:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   14000249e:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   1400024a1:	8b 00                                           	mov    eax,DWORD PTR [rax]
   1400024a3:	25 ff ff ff 20                                  	and    eax,0x20ffffff
   1400024a8:	3d 43 43 47 20                                  	cmp    eax,0x20474343
   1400024ad:	75 1b                                           	jne    1400024ca <_gnu_exception_handler+0x4a>
   1400024af:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   1400024b3:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   1400024b6:	8b 40 04                                        	mov    eax,DWORD PTR [rax+0x4]
   1400024b9:	83 e0 01                                        	and    eax,0x1
   1400024bc:	85 c0                                           	test   eax,eax
   1400024be:	75 0a                                           	jne    1400024ca <_gnu_exception_handler+0x4a>
   1400024c0:	b8 ff ff ff ff                                  	mov    eax,0xffffffff
   1400024c5:	e9 d3 01 00 00                                  	jmp    14000269d <_gnu_exception_handler+0x21d>
   1400024ca:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   1400024ce:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   1400024d1:	8b 00                                           	mov    eax,DWORD PTR [rax]
   1400024d3:	3d 96 00 00 c0                                  	cmp    eax,0xc0000096
   1400024d8:	0f 87 8d 01 00 00                               	ja     14000266b <_gnu_exception_handler+0x1eb>
   1400024de:	3d 8c 00 00 c0                                  	cmp    eax,0xc000008c
   1400024e3:	73 43                                           	jae    140002528 <_gnu_exception_handler+0xa8>
   1400024e5:	3d 1d 00 00 c0                                  	cmp    eax,0xc000001d
   1400024ea:	0f 84 bf 00 00 00                               	je     1400025af <_gnu_exception_handler+0x12f>
   1400024f0:	3d 1d 00 00 c0                                  	cmp    eax,0xc000001d
   1400024f5:	0f 87 70 01 00 00                               	ja     14000266b <_gnu_exception_handler+0x1eb>
   1400024fb:	3d 08 00 00 c0                                  	cmp    eax,0xc0000008
   140002500:	0f 84 5c 01 00 00                               	je     140002662 <_gnu_exception_handler+0x1e2>
   140002506:	3d 08 00 00 c0                                  	cmp    eax,0xc0000008
   14000250b:	0f 87 5a 01 00 00                               	ja     14000266b <_gnu_exception_handler+0x1eb>
   140002511:	3d 02 00 00 80                                  	cmp    eax,0x80000002
   140002516:	0f 84 46 01 00 00                               	je     140002662 <_gnu_exception_handler+0x1e2>
   14000251c:	3d 05 00 00 c0                                  	cmp    eax,0xc0000005
   140002521:	74 35                                           	je     140002558 <_gnu_exception_handler+0xd8>
   140002523:	e9 43 01 00 00                                  	jmp    14000266b <_gnu_exception_handler+0x1eb>
   140002528:	05 74 ff ff 3f                                  	add    eax,0x3fffff74
   14000252d:	83 f8 0a                                        	cmp    eax,0xa
   140002530:	0f 87 35 01 00 00                               	ja     14000266b <_gnu_exception_handler+0x1eb>
   140002536:	89 c0                                           	mov    eax,eax
   140002538:	48 8d 14 85 00 00 00 00                         	lea    rdx,[rax*4+0x0]
   140002540:	48 8d 05 f9 2d 00 00                            	lea    rax,[rip+0x2df9]        # 140005340 <.rdata>
   140002547:	8b 04 02                                        	mov    eax,DWORD PTR [rdx+rax*1]
   14000254a:	48 98                                           	cdqe
   14000254c:	48 8d 15 ed 2d 00 00                            	lea    rdx,[rip+0x2ded]        # 140005340 <.rdata>
   140002553:	48 01 d0                                        	add    rax,rdx
   140002556:	ff e0                                           	jmp    rax
   140002558:	ba 00 00 00 00                                  	mov    edx,0x0
   14000255d:	b9 0b 00 00 00                                  	mov    ecx,0xb
   140002562:	e8 51 0d 00 00                                  	call   1400032b8 <signal>
   140002567:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   14000256b:	48 83 7d f0 01                                  	cmp    QWORD PTR [rbp-0x10],0x1
   140002570:	75 1b                                           	jne    14000258d <_gnu_exception_handler+0x10d>
   140002572:	ba 01 00 00 00                                  	mov    edx,0x1
   140002577:	b9 0b 00 00 00                                  	mov    ecx,0xb
   14000257c:	e8 37 0d 00 00                                  	call   1400032b8 <signal>
   140002581:	c7 45 fc ff ff ff ff                            	mov    DWORD PTR [rbp-0x4],0xffffffff
   140002588:	e9 e1 00 00 00                                  	jmp    14000266e <_gnu_exception_handler+0x1ee>
   14000258d:	48 83 7d f0 00                                  	cmp    QWORD PTR [rbp-0x10],0x0
   140002592:	0f 84 d6 00 00 00                               	je     14000266e <_gnu_exception_handler+0x1ee>
   140002598:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   14000259c:	b9 0b 00 00 00                                  	mov    ecx,0xb
   1400025a1:	ff d0                                           	call   rax
   1400025a3:	c7 45 fc ff ff ff ff                            	mov    DWORD PTR [rbp-0x4],0xffffffff
   1400025aa:	e9 bf 00 00 00                                  	jmp    14000266e <_gnu_exception_handler+0x1ee>
   1400025af:	ba 00 00 00 00                                  	mov    edx,0x0
   1400025b4:	b9 04 00 00 00                                  	mov    ecx,0x4
   1400025b9:	e8 fa 0c 00 00                                  	call   1400032b8 <signal>
   1400025be:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   1400025c2:	48 83 7d f0 01                                  	cmp    QWORD PTR [rbp-0x10],0x1
   1400025c7:	75 1b                                           	jne    1400025e4 <_gnu_exception_handler+0x164>
   1400025c9:	ba 01 00 00 00                                  	mov    edx,0x1
   1400025ce:	b9 04 00 00 00                                  	mov    ecx,0x4
   1400025d3:	e8 e0 0c 00 00                                  	call   1400032b8 <signal>
   1400025d8:	c7 45 fc ff ff ff ff                            	mov    DWORD PTR [rbp-0x4],0xffffffff
   1400025df:	e9 8d 00 00 00                                  	jmp    140002671 <_gnu_exception_handler+0x1f1>
   1400025e4:	48 83 7d f0 00                                  	cmp    QWORD PTR [rbp-0x10],0x0
   1400025e9:	0f 84 82 00 00 00                               	je     140002671 <_gnu_exception_handler+0x1f1>
   1400025ef:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400025f3:	b9 04 00 00 00                                  	mov    ecx,0x4
   1400025f8:	ff d0                                           	call   rax
   1400025fa:	c7 45 fc ff ff ff ff                            	mov    DWORD PTR [rbp-0x4],0xffffffff
   140002601:	eb 6e                                           	jmp    140002671 <_gnu_exception_handler+0x1f1>
   140002603:	c7 45 f8 01 00 00 00                            	mov    DWORD PTR [rbp-0x8],0x1
   14000260a:	ba 00 00 00 00                                  	mov    edx,0x0
   14000260f:	b9 08 00 00 00                                  	mov    ecx,0x8
   140002614:	e8 9f 0c 00 00                                  	call   1400032b8 <signal>
   140002619:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   14000261d:	48 83 7d f0 01                                  	cmp    QWORD PTR [rbp-0x10],0x1
   140002622:	75 23                                           	jne    140002647 <_gnu_exception_handler+0x1c7>
   140002624:	ba 01 00 00 00                                  	mov    edx,0x1
   140002629:	b9 08 00 00 00                                  	mov    ecx,0x8
   14000262e:	e8 85 0c 00 00                                  	call   1400032b8 <signal>
   140002633:	83 7d f8 00                                     	cmp    DWORD PTR [rbp-0x8],0x0
   140002637:	74 05                                           	je     14000263e <_gnu_exception_handler+0x1be>
   140002639:	e8 f2 f4 ff ff                                  	call   140001b30 <_fpreset>
   14000263e:	c7 45 fc ff ff ff ff                            	mov    DWORD PTR [rbp-0x4],0xffffffff
   140002645:	eb 2d                                           	jmp    140002674 <_gnu_exception_handler+0x1f4>
   140002647:	48 83 7d f0 00                                  	cmp    QWORD PTR [rbp-0x10],0x0
   14000264c:	74 26                                           	je     140002674 <_gnu_exception_handler+0x1f4>
   14000264e:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140002652:	b9 08 00 00 00                                  	mov    ecx,0x8
   140002657:	ff d0                                           	call   rax
   140002659:	c7 45 fc ff ff ff ff                            	mov    DWORD PTR [rbp-0x4],0xffffffff
   140002660:	eb 12                                           	jmp    140002674 <_gnu_exception_handler+0x1f4>
   140002662:	c7 45 fc ff ff ff ff                            	mov    DWORD PTR [rbp-0x4],0xffffffff
   140002669:	eb 0a                                           	jmp    140002675 <_gnu_exception_handler+0x1f5>
   14000266b:	90                                              	nop
   14000266c:	eb 07                                           	jmp    140002675 <_gnu_exception_handler+0x1f5>
   14000266e:	90                                              	nop
   14000266f:	eb 04                                           	jmp    140002675 <_gnu_exception_handler+0x1f5>
   140002671:	90                                              	nop
   140002672:	eb 01                                           	jmp    140002675 <_gnu_exception_handler+0x1f5>
   140002674:	90                                              	nop
   140002675:	83 7d fc 00                                     	cmp    DWORD PTR [rbp-0x4],0x0
   140002679:	75 1f                                           	jne    14000269a <_gnu_exception_handler+0x21a>
   14000267b:	48 8b 05 9e 6a 00 00                            	mov    rax,QWORD PTR [rip+0x6a9e]        # 140009120 <__mingw_oldexcpt_handler>
   140002682:	48 85 c0                                        	test   rax,rax
   140002685:	74 13                                           	je     14000269a <_gnu_exception_handler+0x21a>
   140002687:	48 8b 15 92 6a 00 00                            	mov    rdx,QWORD PTR [rip+0x6a92]        # 140009120 <__mingw_oldexcpt_handler>
   14000268e:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140002692:	48 89 c1                                        	mov    rcx,rax
   140002695:	ff d2                                           	call   rdx
   140002697:	89 45 fc                                        	mov    DWORD PTR [rbp-0x4],eax
   14000269a:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   14000269d:	48 83 c4 30                                     	add    rsp,0x30
   1400026a1:	5d                                              	pop    rbp
   1400026a2:	c3                                              	ret
   1400026a3:	90                                              	nop
   1400026a4:	90                                              	nop
   1400026a5:	90                                              	nop
   1400026a6:	90                                              	nop
   1400026a7:	90                                              	nop
   1400026a8:	90                                              	nop
   1400026a9:	90                                              	nop
   1400026aa:	90                                              	nop
   1400026ab:	90                                              	nop
   1400026ac:	90                                              	nop
   1400026ad:	90                                              	nop
   1400026ae:	90                                              	nop
   1400026af:	90                                              	nop

00000001400026b0 <___w64_mingwthr_add_key_dtor>:
   1400026b0:	55                                              	push   rbp
   1400026b1:	48 89 e5                                        	mov    rbp,rsp
   1400026b4:	48 83 ec 30                                     	sub    rsp,0x30
   1400026b8:	89 4d 10                                        	mov    DWORD PTR [rbp+0x10],ecx
   1400026bb:	48 89 55 18                                     	mov    QWORD PTR [rbp+0x18],rdx
   1400026bf:	8b 05 a3 6a 00 00                               	mov    eax,DWORD PTR [rip+0x6aa3]        # 140009168 <__mingwthr_cs_init>
   1400026c5:	85 c0                                           	test   eax,eax
   1400026c7:	75 07                                           	jne    1400026d0 <___w64_mingwthr_add_key_dtor+0x20>
   1400026c9:	b8 00 00 00 00                                  	mov    eax,0x0
   1400026ce:	eb 7b                                           	jmp    14000274b <___w64_mingwthr_add_key_dtor+0x9b>
   1400026d0:	ba 18 00 00 00                                  	mov    edx,0x18
   1400026d5:	b9 01 00 00 00                                  	mov    ecx,0x1
   1400026da:	e8 a1 0b 00 00                                  	call   140003280 <calloc>
   1400026df:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   1400026e3:	48 83 7d f8 00                                  	cmp    QWORD PTR [rbp-0x8],0x0
   1400026e8:	75 07                                           	jne    1400026f1 <___w64_mingwthr_add_key_dtor+0x41>
   1400026ea:	b8 ff ff ff ff                                  	mov    eax,0xffffffff
   1400026ef:	eb 5a                                           	jmp    14000274b <___w64_mingwthr_add_key_dtor+0x9b>
   1400026f1:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   1400026f5:	8b 55 10                                        	mov    edx,DWORD PTR [rbp+0x10]
   1400026f8:	89 10                                           	mov    DWORD PTR [rax],edx
   1400026fa:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   1400026fe:	48 8b 55 18                                     	mov    rdx,QWORD PTR [rbp+0x18]
   140002702:	48 89 50 08                                     	mov    QWORD PTR [rax+0x8],rdx
   140002706:	48 8d 05 33 6a 00 00                            	lea    rax,[rip+0x6a33]        # 140009140 <__mingwthr_cs>
   14000270d:	48 89 c1                                        	mov    rcx,rax
   140002710:	48 8b 05 f1 7a 00 00                            	mov    rax,QWORD PTR [rip+0x7af1]        # 14000a208 <__imp_EnterCriticalSection>
   140002717:	ff d0                                           	call   rax
   140002719:	48 8b 15 50 6a 00 00                            	mov    rdx,QWORD PTR [rip+0x6a50]        # 140009170 <key_dtor_list>
   140002720:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002724:	48 89 50 10                                     	mov    QWORD PTR [rax+0x10],rdx
   140002728:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   14000272c:	48 89 05 3d 6a 00 00                            	mov    QWORD PTR [rip+0x6a3d],rax        # 140009170 <key_dtor_list>
   140002733:	48 8d 05 06 6a 00 00                            	lea    rax,[rip+0x6a06]        # 140009140 <__mingwthr_cs>
   14000273a:	48 89 c1                                        	mov    rcx,rax
   14000273d:	48 8b 05 fc 7a 00 00                            	mov    rax,QWORD PTR [rip+0x7afc]        # 14000a240 <__imp_LeaveCriticalSection>
   140002744:	ff d0                                           	call   rax
   140002746:	b8 00 00 00 00                                  	mov    eax,0x0
   14000274b:	48 83 c4 30                                     	add    rsp,0x30
   14000274f:	5d                                              	pop    rbp
   140002750:	c3                                              	ret

0000000140002751 <___w64_mingwthr_remove_key_dtor>:
   140002751:	55                                              	push   rbp
   140002752:	48 89 e5                                        	mov    rbp,rsp
   140002755:	48 83 ec 30                                     	sub    rsp,0x30
   140002759:	89 4d 10                                        	mov    DWORD PTR [rbp+0x10],ecx
   14000275c:	8b 05 06 6a 00 00                               	mov    eax,DWORD PTR [rip+0x6a06]        # 140009168 <__mingwthr_cs_init>
   140002762:	85 c0                                           	test   eax,eax
   140002764:	75 0a                                           	jne    140002770 <___w64_mingwthr_remove_key_dtor+0x1f>
   140002766:	b8 00 00 00 00                                  	mov    eax,0x0
   14000276b:	e9 9c 00 00 00                                  	jmp    14000280c <___w64_mingwthr_remove_key_dtor+0xbb>
   140002770:	48 8d 05 c9 69 00 00                            	lea    rax,[rip+0x69c9]        # 140009140 <__mingwthr_cs>
   140002777:	48 89 c1                                        	mov    rcx,rax
   14000277a:	48 8b 05 87 7a 00 00                            	mov    rax,QWORD PTR [rip+0x7a87]        # 14000a208 <__imp_EnterCriticalSection>
   140002781:	ff d0                                           	call   rax
   140002783:	48 c7 45 f8 00 00 00 00                         	mov    QWORD PTR [rbp-0x8],0x0
   14000278b:	48 8b 05 de 69 00 00                            	mov    rax,QWORD PTR [rip+0x69de]        # 140009170 <key_dtor_list>
   140002792:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   140002796:	eb 55                                           	jmp    1400027ed <___w64_mingwthr_remove_key_dtor+0x9c>
   140002798:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   14000279c:	8b 00                                           	mov    eax,DWORD PTR [rax]
   14000279e:	39 45 10                                        	cmp    DWORD PTR [rbp+0x10],eax
   1400027a1:	75 36                                           	jne    1400027d9 <___w64_mingwthr_remove_key_dtor+0x88>
   1400027a3:	48 83 7d f8 00                                  	cmp    QWORD PTR [rbp-0x8],0x0
   1400027a8:	75 11                                           	jne    1400027bb <___w64_mingwthr_remove_key_dtor+0x6a>
   1400027aa:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400027ae:	48 8b 40 10                                     	mov    rax,QWORD PTR [rax+0x10]
   1400027b2:	48 89 05 b7 69 00 00                            	mov    QWORD PTR [rip+0x69b7],rax        # 140009170 <key_dtor_list>
   1400027b9:	eb 10                                           	jmp    1400027cb <___w64_mingwthr_remove_key_dtor+0x7a>
   1400027bb:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400027bf:	48 8b 50 10                                     	mov    rdx,QWORD PTR [rax+0x10]
   1400027c3:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   1400027c7:	48 89 50 10                                     	mov    QWORD PTR [rax+0x10],rdx
   1400027cb:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400027cf:	48 89 c1                                        	mov    rcx,rax
   1400027d2:	e8 b9 0a 00 00                                  	call   140003290 <free>
   1400027d7:	eb 1b                                           	jmp    1400027f4 <___w64_mingwthr_remove_key_dtor+0xa3>
   1400027d9:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400027dd:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   1400027e1:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400027e5:	48 8b 40 10                                     	mov    rax,QWORD PTR [rax+0x10]
   1400027e9:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   1400027ed:	48 83 7d f0 00                                  	cmp    QWORD PTR [rbp-0x10],0x0
   1400027f2:	75 a4                                           	jne    140002798 <___w64_mingwthr_remove_key_dtor+0x47>
   1400027f4:	48 8d 05 45 69 00 00                            	lea    rax,[rip+0x6945]        # 140009140 <__mingwthr_cs>
   1400027fb:	48 89 c1                                        	mov    rcx,rax
   1400027fe:	48 8b 05 3b 7a 00 00                            	mov    rax,QWORD PTR [rip+0x7a3b]        # 14000a240 <__imp_LeaveCriticalSection>
   140002805:	ff d0                                           	call   rax
   140002807:	b8 00 00 00 00                                  	mov    eax,0x0
   14000280c:	48 83 c4 30                                     	add    rsp,0x30
   140002810:	5d                                              	pop    rbp
   140002811:	c3                                              	ret

0000000140002812 <__mingwthr_run_key_dtors>:
   140002812:	55                                              	push   rbp
   140002813:	48 89 e5                                        	mov    rbp,rsp
   140002816:	48 83 ec 30                                     	sub    rsp,0x30
   14000281a:	8b 05 48 69 00 00                               	mov    eax,DWORD PTR [rip+0x6948]        # 140009168 <__mingwthr_cs_init>
   140002820:	85 c0                                           	test   eax,eax
   140002822:	0f 84 82 00 00 00                               	je     1400028aa <__mingwthr_run_key_dtors+0x98>
   140002828:	48 8d 05 11 69 00 00                            	lea    rax,[rip+0x6911]        # 140009140 <__mingwthr_cs>
   14000282f:	48 89 c1                                        	mov    rcx,rax
   140002832:	48 8b 05 cf 79 00 00                            	mov    rax,QWORD PTR [rip+0x79cf]        # 14000a208 <__imp_EnterCriticalSection>
   140002839:	ff d0                                           	call   rax
   14000283b:	48 8b 05 2e 69 00 00                            	mov    rax,QWORD PTR [rip+0x692e]        # 140009170 <key_dtor_list>
   140002842:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140002846:	eb 46                                           	jmp    14000288e <__mingwthr_run_key_dtors+0x7c>
   140002848:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   14000284c:	8b 00                                           	mov    eax,DWORD PTR [rax]
   14000284e:	89 c1                                           	mov    ecx,eax
   140002850:	48 8b 05 09 7a 00 00                            	mov    rax,QWORD PTR [rip+0x7a09]        # 14000a260 <__imp_TlsGetValue>
   140002857:	ff d0                                           	call   rax
   140002859:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   14000285d:	48 8b 05 b4 79 00 00                            	mov    rax,QWORD PTR [rip+0x79b4]        # 14000a218 <__imp_GetLastError>
   140002864:	ff d0                                           	call   rax
   140002866:	85 c0                                           	test   eax,eax
   140002868:	75 18                                           	jne    140002882 <__mingwthr_run_key_dtors+0x70>
   14000286a:	48 83 7d f0 00                                  	cmp    QWORD PTR [rbp-0x10],0x0
   14000286f:	74 11                                           	je     140002882 <__mingwthr_run_key_dtors+0x70>
   140002871:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002875:	48 8b 50 08                                     	mov    rdx,QWORD PTR [rax+0x8]
   140002879:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   14000287d:	48 89 c1                                        	mov    rcx,rax
   140002880:	ff d2                                           	call   rdx
   140002882:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002886:	48 8b 40 10                                     	mov    rax,QWORD PTR [rax+0x10]
   14000288a:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   14000288e:	48 83 7d f8 00                                  	cmp    QWORD PTR [rbp-0x8],0x0
   140002893:	75 b3                                           	jne    140002848 <__mingwthr_run_key_dtors+0x36>
   140002895:	48 8d 05 a4 68 00 00                            	lea    rax,[rip+0x68a4]        # 140009140 <__mingwthr_cs>
   14000289c:	48 89 c1                                        	mov    rcx,rax
   14000289f:	48 8b 05 9a 79 00 00                            	mov    rax,QWORD PTR [rip+0x799a]        # 14000a240 <__imp_LeaveCriticalSection>
   1400028a6:	ff d0                                           	call   rax
   1400028a8:	eb 01                                           	jmp    1400028ab <__mingwthr_run_key_dtors+0x99>
   1400028aa:	90                                              	nop
   1400028ab:	48 83 c4 30                                     	add    rsp,0x30
   1400028af:	5d                                              	pop    rbp
   1400028b0:	c3                                              	ret

00000001400028b1 <__mingw_TLScallback>:
   1400028b1:	55                                              	push   rbp
   1400028b2:	48 89 e5                                        	mov    rbp,rsp
   1400028b5:	48 83 ec 30                                     	sub    rsp,0x30
   1400028b9:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   1400028bd:	89 55 18                                        	mov    DWORD PTR [rbp+0x18],edx
   1400028c0:	4c 89 45 20                                     	mov    QWORD PTR [rbp+0x20],r8
   1400028c4:	83 7d 18 03                                     	cmp    DWORD PTR [rbp+0x18],0x3
   1400028c8:	0f 84 cc 00 00 00                               	je     14000299a <__mingw_TLScallback+0xe9>
   1400028ce:	83 7d 18 03                                     	cmp    DWORD PTR [rbp+0x18],0x3
   1400028d2:	0f 87 ca 00 00 00                               	ja     1400029a2 <__mingw_TLScallback+0xf1>
   1400028d8:	83 7d 18 02                                     	cmp    DWORD PTR [rbp+0x18],0x2
   1400028dc:	0f 84 b1 00 00 00                               	je     140002993 <__mingw_TLScallback+0xe2>
   1400028e2:	83 7d 18 02                                     	cmp    DWORD PTR [rbp+0x18],0x2
   1400028e6:	0f 87 b6 00 00 00                               	ja     1400029a2 <__mingw_TLScallback+0xf1>
   1400028ec:	83 7d 18 00                                     	cmp    DWORD PTR [rbp+0x18],0x0
   1400028f0:	74 33                                           	je     140002925 <__mingw_TLScallback+0x74>
   1400028f2:	83 7d 18 01                                     	cmp    DWORD PTR [rbp+0x18],0x1
   1400028f6:	0f 85 a6 00 00 00                               	jne    1400029a2 <__mingw_TLScallback+0xf1>
   1400028fc:	8b 05 66 68 00 00                               	mov    eax,DWORD PTR [rip+0x6866]        # 140009168 <__mingwthr_cs_init>
   140002902:	85 c0                                           	test   eax,eax
   140002904:	75 13                                           	jne    140002919 <__mingw_TLScallback+0x68>
   140002906:	48 8d 05 33 68 00 00                            	lea    rax,[rip+0x6833]        # 140009140 <__mingwthr_cs>
   14000290d:	48 89 c1                                        	mov    rcx,rax
   140002910:	48 8b 05 21 79 00 00                            	mov    rax,QWORD PTR [rip+0x7921]        # 14000a238 <__imp_InitializeCriticalSection>
   140002917:	ff d0                                           	call   rax
   140002919:	c7 05 45 68 00 00 01 00 00 00                   	mov    DWORD PTR [rip+0x6845],0x1        # 140009168 <__mingwthr_cs_init>
   140002923:	eb 7d                                           	jmp    1400029a2 <__mingw_TLScallback+0xf1>
   140002925:	e8 e8 fe ff ff                                  	call   140002812 <__mingwthr_run_key_dtors>
   14000292a:	8b 05 38 68 00 00                               	mov    eax,DWORD PTR [rip+0x6838]        # 140009168 <__mingwthr_cs_init>
   140002930:	83 f8 01                                        	cmp    eax,0x1
   140002933:	75 6c                                           	jne    1400029a1 <__mingw_TLScallback+0xf0>
   140002935:	48 8b 05 34 68 00 00                            	mov    rax,QWORD PTR [rip+0x6834]        # 140009170 <key_dtor_list>
   14000293c:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140002940:	eb 20                                           	jmp    140002962 <__mingw_TLScallback+0xb1>
   140002942:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002946:	48 8b 40 10                                     	mov    rax,QWORD PTR [rax+0x10]
   14000294a:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   14000294e:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002952:	48 89 c1                                        	mov    rcx,rax
   140002955:	e8 36 09 00 00                                  	call   140003290 <free>
   14000295a:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   14000295e:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140002962:	48 83 7d f8 00                                  	cmp    QWORD PTR [rbp-0x8],0x0
   140002967:	75 d9                                           	jne    140002942 <__mingw_TLScallback+0x91>
   140002969:	48 c7 05 fc 67 00 00 00 00 00 00                	mov    QWORD PTR [rip+0x67fc],0x0        # 140009170 <key_dtor_list>
   140002974:	c7 05 ea 67 00 00 00 00 00 00                   	mov    DWORD PTR [rip+0x67ea],0x0        # 140009168 <__mingwthr_cs_init>
   14000297e:	48 8d 05 bb 67 00 00                            	lea    rax,[rip+0x67bb]        # 140009140 <__mingwthr_cs>
   140002985:	48 89 c1                                        	mov    rcx,rax
   140002988:	48 8b 05 71 78 00 00                            	mov    rax,QWORD PTR [rip+0x7871]        # 14000a200 <__IAT_start__>
   14000298f:	ff d0                                           	call   rax
   140002991:	eb 0e                                           	jmp    1400029a1 <__mingw_TLScallback+0xf0>
   140002993:	e8 98 f1 ff ff                                  	call   140001b30 <_fpreset>
   140002998:	eb 08                                           	jmp    1400029a2 <__mingw_TLScallback+0xf1>
   14000299a:	e8 73 fe ff ff                                  	call   140002812 <__mingwthr_run_key_dtors>
   14000299f:	eb 01                                           	jmp    1400029a2 <__mingw_TLScallback+0xf1>
   1400029a1:	90                                              	nop
   1400029a2:	b8 01 00 00 00                                  	mov    eax,0x1
   1400029a7:	48 83 c4 30                                     	add    rsp,0x30
   1400029ab:	5d                                              	pop    rbp
   1400029ac:	c3                                              	ret
   1400029ad:	90                                              	nop
   1400029ae:	90                                              	nop
   1400029af:	90                                              	nop

00000001400029b0 <_ValidateImageBase>:
   1400029b0:	55                                              	push   rbp
   1400029b1:	48 89 e5                                        	mov    rbp,rsp
   1400029b4:	48 83 ec 20                                     	sub    rsp,0x20
   1400029b8:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   1400029bc:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   1400029c0:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   1400029c4:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   1400029c8:	0f b7 00                                        	movzx  eax,WORD PTR [rax]
   1400029cb:	66 3d 4d 5a                                     	cmp    ax,0x5a4d
   1400029cf:	74 07                                           	je     1400029d8 <_ValidateImageBase+0x28>
   1400029d1:	b8 00 00 00 00                                  	mov    eax,0x0
   1400029d6:	eb 4e                                           	jmp    140002a26 <_ValidateImageBase+0x76>
   1400029d8:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   1400029dc:	8b 40 3c                                        	mov    eax,DWORD PTR [rax+0x3c]
   1400029df:	48 63 d0                                        	movsxd rdx,eax
   1400029e2:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   1400029e6:	48 01 d0                                        	add    rax,rdx
   1400029e9:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   1400029ed:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   1400029f1:	8b 00                                           	mov    eax,DWORD PTR [rax]
   1400029f3:	3d 50 45 00 00                                  	cmp    eax,0x4550
   1400029f8:	74 07                                           	je     140002a01 <_ValidateImageBase+0x51>
   1400029fa:	b8 00 00 00 00                                  	mov    eax,0x0
   1400029ff:	eb 25                                           	jmp    140002a26 <_ValidateImageBase+0x76>
   140002a01:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140002a05:	48 83 c0 18                                     	add    rax,0x18
   140002a09:	48 89 45 e8                                     	mov    QWORD PTR [rbp-0x18],rax
   140002a0d:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002a11:	0f b7 00                                        	movzx  eax,WORD PTR [rax]
   140002a14:	66 3d 0b 02                                     	cmp    ax,0x20b
   140002a18:	74 07                                           	je     140002a21 <_ValidateImageBase+0x71>
   140002a1a:	b8 00 00 00 00                                  	mov    eax,0x0
   140002a1f:	eb 05                                           	jmp    140002a26 <_ValidateImageBase+0x76>
   140002a21:	b8 01 00 00 00                                  	mov    eax,0x1
   140002a26:	48 83 c4 20                                     	add    rsp,0x20
   140002a2a:	5d                                              	pop    rbp
   140002a2b:	c3                                              	ret

0000000140002a2c <_FindPESection>:
   140002a2c:	55                                              	push   rbp
   140002a2d:	48 89 e5                                        	mov    rbp,rsp
   140002a30:	48 83 ec 20                                     	sub    rsp,0x20
   140002a34:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140002a38:	48 89 55 18                                     	mov    QWORD PTR [rbp+0x18],rdx
   140002a3c:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140002a40:	8b 40 3c                                        	mov    eax,DWORD PTR [rax+0x3c]
   140002a43:	48 63 d0                                        	movsxd rdx,eax
   140002a46:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140002a4a:	48 01 d0                                        	add    rax,rdx
   140002a4d:	48 89 45 e8                                     	mov    QWORD PTR [rbp-0x18],rax
   140002a51:	c7 45 f4 00 00 00 00                            	mov    DWORD PTR [rbp-0xc],0x0
   140002a58:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002a5c:	0f b7 40 14                                     	movzx  eax,WORD PTR [rax+0x14]
   140002a60:	0f b7 d0                                        	movzx  edx,ax
   140002a63:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002a67:	48 01 d0                                        	add    rax,rdx
   140002a6a:	48 83 c0 18                                     	add    rax,0x18
   140002a6e:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140002a72:	eb 36                                           	jmp    140002aaa <_FindPESection+0x7e>
   140002a74:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002a78:	8b 40 0c                                        	mov    eax,DWORD PTR [rax+0xc]
   140002a7b:	89 c0                                           	mov    eax,eax
   140002a7d:	48 39 45 18                                     	cmp    QWORD PTR [rbp+0x18],rax
   140002a81:	72 1e                                           	jb     140002aa1 <_FindPESection+0x75>
   140002a83:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002a87:	8b 50 0c                                        	mov    edx,DWORD PTR [rax+0xc]
   140002a8a:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002a8e:	8b 40 08                                        	mov    eax,DWORD PTR [rax+0x8]
   140002a91:	01 d0                                           	add    eax,edx
   140002a93:	89 c0                                           	mov    eax,eax
   140002a95:	48 39 45 18                                     	cmp    QWORD PTR [rbp+0x18],rax
   140002a99:	73 06                                           	jae    140002aa1 <_FindPESection+0x75>
   140002a9b:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002a9f:	eb 1e                                           	jmp    140002abf <_FindPESection+0x93>
   140002aa1:	83 45 f4 01                                     	add    DWORD PTR [rbp-0xc],0x1
   140002aa5:	48 83 45 f8 28                                  	add    QWORD PTR [rbp-0x8],0x28
   140002aaa:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002aae:	0f b7 40 06                                     	movzx  eax,WORD PTR [rax+0x6]
   140002ab2:	0f b7 c0                                        	movzx  eax,ax
   140002ab5:	39 45 f4                                        	cmp    DWORD PTR [rbp-0xc],eax
   140002ab8:	72 ba                                           	jb     140002a74 <_FindPESection+0x48>
   140002aba:	b8 00 00 00 00                                  	mov    eax,0x0
   140002abf:	48 83 c4 20                                     	add    rsp,0x20
   140002ac3:	5d                                              	pop    rbp
   140002ac4:	c3                                              	ret

0000000140002ac5 <_FindPESectionByName>:
   140002ac5:	55                                              	push   rbp
   140002ac6:	48 89 e5                                        	mov    rbp,rsp
   140002ac9:	48 83 ec 40                                     	sub    rsp,0x40
   140002acd:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140002ad1:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140002ad5:	48 89 c1                                        	mov    rcx,rax
   140002ad8:	e8 e3 07 00 00                                  	call   1400032c0 <strlen>
   140002add:	48 83 f8 08                                     	cmp    rax,0x8
   140002ae1:	76 0a                                           	jbe    140002aed <_FindPESectionByName+0x28>
   140002ae3:	b8 00 00 00 00                                  	mov    eax,0x0
   140002ae8:	e9 98 00 00 00                                  	jmp    140002b85 <_FindPESectionByName+0xc0>
   140002aed:	48 8b 05 0c 29 00 00                            	mov    rax,QWORD PTR [rip+0x290c]        # 140005400 <.refptr.__image_base__>
   140002af4:	48 89 45 e8                                     	mov    QWORD PTR [rbp-0x18],rax
   140002af8:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002afc:	48 89 c1                                        	mov    rcx,rax
   140002aff:	e8 ac fe ff ff                                  	call   1400029b0 <_ValidateImageBase>
   140002b04:	85 c0                                           	test   eax,eax
   140002b06:	75 07                                           	jne    140002b0f <_FindPESectionByName+0x4a>
   140002b08:	b8 00 00 00 00                                  	mov    eax,0x0
   140002b0d:	eb 76                                           	jmp    140002b85 <_FindPESectionByName+0xc0>
   140002b0f:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002b13:	8b 40 3c                                        	mov    eax,DWORD PTR [rax+0x3c]
   140002b16:	48 63 d0                                        	movsxd rdx,eax
   140002b19:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002b1d:	48 01 d0                                        	add    rax,rdx
   140002b20:	48 89 45 e0                                     	mov    QWORD PTR [rbp-0x20],rax
   140002b24:	c7 45 f4 00 00 00 00                            	mov    DWORD PTR [rbp-0xc],0x0
   140002b2b:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140002b2f:	0f b7 40 14                                     	movzx  eax,WORD PTR [rax+0x14]
   140002b33:	0f b7 d0                                        	movzx  edx,ax
   140002b36:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140002b3a:	48 01 d0                                        	add    rax,rdx
   140002b3d:	48 83 c0 18                                     	add    rax,0x18
   140002b41:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140002b45:	eb 29                                           	jmp    140002b70 <_FindPESectionByName+0xab>
   140002b47:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002b4b:	48 8b 55 10                                     	mov    rdx,QWORD PTR [rbp+0x10]
   140002b4f:	41 b8 08 00 00 00                               	mov    r8d,0x8
   140002b55:	48 89 c1                                        	mov    rcx,rax
   140002b58:	e8 6b 07 00 00                                  	call   1400032c8 <strncmp>
   140002b5d:	85 c0                                           	test   eax,eax
   140002b5f:	75 06                                           	jne    140002b67 <_FindPESectionByName+0xa2>
   140002b61:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002b65:	eb 1e                                           	jmp    140002b85 <_FindPESectionByName+0xc0>
   140002b67:	83 45 f4 01                                     	add    DWORD PTR [rbp-0xc],0x1
   140002b6b:	48 83 45 f8 28                                  	add    QWORD PTR [rbp-0x8],0x28
   140002b70:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140002b74:	0f b7 40 06                                     	movzx  eax,WORD PTR [rax+0x6]
   140002b78:	0f b7 c0                                        	movzx  eax,ax
   140002b7b:	39 45 f4                                        	cmp    DWORD PTR [rbp-0xc],eax
   140002b7e:	72 c7                                           	jb     140002b47 <_FindPESectionByName+0x82>
   140002b80:	b8 00 00 00 00                                  	mov    eax,0x0
   140002b85:	48 83 c4 40                                     	add    rsp,0x40
   140002b89:	5d                                              	pop    rbp
   140002b8a:	c3                                              	ret

0000000140002b8b <__mingw_GetSectionForAddress>:
   140002b8b:	55                                              	push   rbp
   140002b8c:	48 89 e5                                        	mov    rbp,rsp
   140002b8f:	48 83 ec 30                                     	sub    rsp,0x30
   140002b93:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140002b97:	48 8b 05 62 28 00 00                            	mov    rax,QWORD PTR [rip+0x2862]        # 140005400 <.refptr.__image_base__>
   140002b9e:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140002ba2:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002ba6:	48 89 c1                                        	mov    rcx,rax
   140002ba9:	e8 02 fe ff ff                                  	call   1400029b0 <_ValidateImageBase>
   140002bae:	85 c0                                           	test   eax,eax
   140002bb0:	75 07                                           	jne    140002bb9 <__mingw_GetSectionForAddress+0x2e>
   140002bb2:	b8 00 00 00 00                                  	mov    eax,0x0
   140002bb7:	eb 1c                                           	jmp    140002bd5 <__mingw_GetSectionForAddress+0x4a>
   140002bb9:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140002bbd:	48 2b 45 f8                                     	sub    rax,QWORD PTR [rbp-0x8]
   140002bc1:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   140002bc5:	48 8b 55 f0                                     	mov    rdx,QWORD PTR [rbp-0x10]
   140002bc9:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002bcd:	48 89 c1                                        	mov    rcx,rax
   140002bd0:	e8 57 fe ff ff                                  	call   140002a2c <_FindPESection>
   140002bd5:	48 83 c4 30                                     	add    rsp,0x30
   140002bd9:	5d                                              	pop    rbp
   140002bda:	c3                                              	ret

0000000140002bdb <__mingw_GetSectionCount>:
   140002bdb:	55                                              	push   rbp
   140002bdc:	48 89 e5                                        	mov    rbp,rsp
   140002bdf:	48 83 ec 30                                     	sub    rsp,0x30
   140002be3:	48 8b 05 16 28 00 00                            	mov    rax,QWORD PTR [rip+0x2816]        # 140005400 <.refptr.__image_base__>
   140002bea:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140002bee:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002bf2:	48 89 c1                                        	mov    rcx,rax
   140002bf5:	e8 b6 fd ff ff                                  	call   1400029b0 <_ValidateImageBase>
   140002bfa:	85 c0                                           	test   eax,eax
   140002bfc:	75 07                                           	jne    140002c05 <__mingw_GetSectionCount+0x2a>
   140002bfe:	b8 00 00 00 00                                  	mov    eax,0x0
   140002c03:	eb 20                                           	jmp    140002c25 <__mingw_GetSectionCount+0x4a>
   140002c05:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002c09:	8b 40 3c                                        	mov    eax,DWORD PTR [rax+0x3c]
   140002c0c:	48 63 d0                                        	movsxd rdx,eax
   140002c0f:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002c13:	48 01 d0                                        	add    rax,rdx
   140002c16:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   140002c1a:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140002c1e:	0f b7 40 06                                     	movzx  eax,WORD PTR [rax+0x6]
   140002c22:	0f b7 c0                                        	movzx  eax,ax
   140002c25:	48 83 c4 30                                     	add    rsp,0x30
   140002c29:	5d                                              	pop    rbp
   140002c2a:	c3                                              	ret

0000000140002c2b <_FindPESectionExec>:
   140002c2b:	55                                              	push   rbp
   140002c2c:	48 89 e5                                        	mov    rbp,rsp
   140002c2f:	48 83 ec 40                                     	sub    rsp,0x40
   140002c33:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140002c37:	48 8b 05 c2 27 00 00                            	mov    rax,QWORD PTR [rip+0x27c2]        # 140005400 <.refptr.__image_base__>
   140002c3e:	48 89 45 e8                                     	mov    QWORD PTR [rbp-0x18],rax
   140002c42:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002c46:	48 89 c1                                        	mov    rcx,rax
   140002c49:	e8 62 fd ff ff                                  	call   1400029b0 <_ValidateImageBase>
   140002c4e:	85 c0                                           	test   eax,eax
   140002c50:	75 07                                           	jne    140002c59 <_FindPESectionExec+0x2e>
   140002c52:	b8 00 00 00 00                                  	mov    eax,0x0
   140002c57:	eb 78                                           	jmp    140002cd1 <_FindPESectionExec+0xa6>
   140002c59:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002c5d:	8b 40 3c                                        	mov    eax,DWORD PTR [rax+0x3c]
   140002c60:	48 63 d0                                        	movsxd rdx,eax
   140002c63:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002c67:	48 01 d0                                        	add    rax,rdx
   140002c6a:	48 89 45 e0                                     	mov    QWORD PTR [rbp-0x20],rax
   140002c6e:	c7 45 f4 00 00 00 00                            	mov    DWORD PTR [rbp-0xc],0x0
   140002c75:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140002c79:	0f b7 40 14                                     	movzx  eax,WORD PTR [rax+0x14]
   140002c7d:	0f b7 d0                                        	movzx  edx,ax
   140002c80:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140002c84:	48 01 d0                                        	add    rax,rdx
   140002c87:	48 83 c0 18                                     	add    rax,0x18
   140002c8b:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140002c8f:	eb 2b                                           	jmp    140002cbc <_FindPESectionExec+0x91>
   140002c91:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002c95:	8b 40 24                                        	mov    eax,DWORD PTR [rax+0x24]
   140002c98:	25 00 00 00 20                                  	and    eax,0x20000000
   140002c9d:	85 c0                                           	test   eax,eax
   140002c9f:	74 12                                           	je     140002cb3 <_FindPESectionExec+0x88>
   140002ca1:	48 83 7d 10 00                                  	cmp    QWORD PTR [rbp+0x10],0x0
   140002ca6:	75 06                                           	jne    140002cae <_FindPESectionExec+0x83>
   140002ca8:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002cac:	eb 23                                           	jmp    140002cd1 <_FindPESectionExec+0xa6>
   140002cae:	48 83 6d 10 01                                  	sub    QWORD PTR [rbp+0x10],0x1
   140002cb3:	83 45 f4 01                                     	add    DWORD PTR [rbp-0xc],0x1
   140002cb7:	48 83 45 f8 28                                  	add    QWORD PTR [rbp-0x8],0x28
   140002cbc:	48 8b 45 e0                                     	mov    rax,QWORD PTR [rbp-0x20]
   140002cc0:	0f b7 40 06                                     	movzx  eax,WORD PTR [rax+0x6]
   140002cc4:	0f b7 c0                                        	movzx  eax,ax
   140002cc7:	39 45 f4                                        	cmp    DWORD PTR [rbp-0xc],eax
   140002cca:	72 c5                                           	jb     140002c91 <_FindPESectionExec+0x66>
   140002ccc:	b8 00 00 00 00                                  	mov    eax,0x0
   140002cd1:	48 83 c4 40                                     	add    rsp,0x40
   140002cd5:	5d                                              	pop    rbp
   140002cd6:	c3                                              	ret

0000000140002cd7 <_GetPEImageBase>:
   140002cd7:	55                                              	push   rbp
   140002cd8:	48 89 e5                                        	mov    rbp,rsp
   140002cdb:	48 83 ec 30                                     	sub    rsp,0x30
   140002cdf:	48 8b 05 1a 27 00 00                            	mov    rax,QWORD PTR [rip+0x271a]        # 140005400 <.refptr.__image_base__>
   140002ce6:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140002cea:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002cee:	48 89 c1                                        	mov    rcx,rax
   140002cf1:	e8 ba fc ff ff                                  	call   1400029b0 <_ValidateImageBase>
   140002cf6:	85 c0                                           	test   eax,eax
   140002cf8:	75 07                                           	jne    140002d01 <_GetPEImageBase+0x2a>
   140002cfa:	b8 00 00 00 00                                  	mov    eax,0x0
   140002cff:	eb 04                                           	jmp    140002d05 <_GetPEImageBase+0x2e>
   140002d01:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002d05:	48 83 c4 30                                     	add    rsp,0x30
   140002d09:	5d                                              	pop    rbp
   140002d0a:	c3                                              	ret

0000000140002d0b <_IsNonwritableInCurrentImage>:
   140002d0b:	55                                              	push   rbp
   140002d0c:	48 89 e5                                        	mov    rbp,rsp
   140002d0f:	48 83 ec 40                                     	sub    rsp,0x40
   140002d13:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140002d17:	48 8b 05 e2 26 00 00                            	mov    rax,QWORD PTR [rip+0x26e2]        # 140005400 <.refptr.__image_base__>
   140002d1e:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140002d22:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002d26:	48 89 c1                                        	mov    rcx,rax
   140002d29:	e8 82 fc ff ff                                  	call   1400029b0 <_ValidateImageBase>
   140002d2e:	85 c0                                           	test   eax,eax
   140002d30:	75 07                                           	jne    140002d39 <_IsNonwritableInCurrentImage+0x2e>
   140002d32:	b8 00 00 00 00                                  	mov    eax,0x0
   140002d37:	eb 3d                                           	jmp    140002d76 <_IsNonwritableInCurrentImage+0x6b>
   140002d39:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140002d3d:	48 2b 45 f8                                     	sub    rax,QWORD PTR [rbp-0x8]
   140002d41:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   140002d45:	48 8b 55 f0                                     	mov    rdx,QWORD PTR [rbp-0x10]
   140002d49:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002d4d:	48 89 c1                                        	mov    rcx,rax
   140002d50:	e8 d7 fc ff ff                                  	call   140002a2c <_FindPESection>
   140002d55:	48 89 45 e8                                     	mov    QWORD PTR [rbp-0x18],rax
   140002d59:	48 83 7d e8 00                                  	cmp    QWORD PTR [rbp-0x18],0x0
   140002d5e:	75 07                                           	jne    140002d67 <_IsNonwritableInCurrentImage+0x5c>
   140002d60:	b8 00 00 00 00                                  	mov    eax,0x0
   140002d65:	eb 0f                                           	jmp    140002d76 <_IsNonwritableInCurrentImage+0x6b>
   140002d67:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002d6b:	8b 40 24                                        	mov    eax,DWORD PTR [rax+0x24]
   140002d6e:	f7 d0                                           	not    eax
   140002d70:	c1 e8 1f                                        	shr    eax,0x1f
   140002d73:	0f b6 c0                                        	movzx  eax,al
   140002d76:	48 83 c4 40                                     	add    rsp,0x40
   140002d7a:	5d                                              	pop    rbp
   140002d7b:	c3                                              	ret

0000000140002d7c <__mingw_enum_import_library_names>:
   140002d7c:	55                                              	push   rbp
   140002d7d:	48 89 e5                                        	mov    rbp,rsp
   140002d80:	48 83 ec 50                                     	sub    rsp,0x50
   140002d84:	89 4d 10                                        	mov    DWORD PTR [rbp+0x10],ecx
   140002d87:	48 8b 05 72 26 00 00                            	mov    rax,QWORD PTR [rip+0x2672]        # 140005400 <.refptr.__image_base__>
   140002d8e:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   140002d92:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140002d96:	48 89 c1                                        	mov    rcx,rax
   140002d99:	e8 12 fc ff ff                                  	call   1400029b0 <_ValidateImageBase>
   140002d9e:	85 c0                                           	test   eax,eax
   140002da0:	75 0a                                           	jne    140002dac <__mingw_enum_import_library_names+0x30>
   140002da2:	b8 00 00 00 00                                  	mov    eax,0x0
   140002da7:	e9 ab 00 00 00                                  	jmp    140002e57 <__mingw_enum_import_library_names+0xdb>
   140002dac:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140002db0:	8b 40 3c                                        	mov    eax,DWORD PTR [rax+0x3c]
   140002db3:	48 63 d0                                        	movsxd rdx,eax
   140002db6:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140002dba:	48 01 d0                                        	add    rax,rdx
   140002dbd:	48 89 45 e8                                     	mov    QWORD PTR [rbp-0x18],rax
   140002dc1:	48 8b 45 e8                                     	mov    rax,QWORD PTR [rbp-0x18]
   140002dc5:	8b 80 90 00 00 00                               	mov    eax,DWORD PTR [rax+0x90]
   140002dcb:	89 45 e4                                        	mov    DWORD PTR [rbp-0x1c],eax
   140002dce:	83 7d e4 00                                     	cmp    DWORD PTR [rbp-0x1c],0x0
   140002dd2:	75 07                                           	jne    140002ddb <__mingw_enum_import_library_names+0x5f>
   140002dd4:	b8 00 00 00 00                                  	mov    eax,0x0
   140002dd9:	eb 7c                                           	jmp    140002e57 <__mingw_enum_import_library_names+0xdb>
   140002ddb:	8b 55 e4                                        	mov    edx,DWORD PTR [rbp-0x1c]
   140002dde:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140002de2:	48 89 c1                                        	mov    rcx,rax
   140002de5:	e8 42 fc ff ff                                  	call   140002a2c <_FindPESection>
   140002dea:	48 89 45 d8                                     	mov    QWORD PTR [rbp-0x28],rax
   140002dee:	48 83 7d d8 00                                  	cmp    QWORD PTR [rbp-0x28],0x0
   140002df3:	75 07                                           	jne    140002dfc <__mingw_enum_import_library_names+0x80>
   140002df5:	b8 00 00 00 00                                  	mov    eax,0x0
   140002dfa:	eb 5b                                           	jmp    140002e57 <__mingw_enum_import_library_names+0xdb>
   140002dfc:	8b 55 e4                                        	mov    edx,DWORD PTR [rbp-0x1c]
   140002dff:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140002e03:	48 01 d0                                        	add    rax,rdx
   140002e06:	48 89 45 f8                                     	mov    QWORD PTR [rbp-0x8],rax
   140002e0a:	48 83 7d f8 00                                  	cmp    QWORD PTR [rbp-0x8],0x0
   140002e0f:	75 07                                           	jne    140002e18 <__mingw_enum_import_library_names+0x9c>
   140002e11:	b8 00 00 00 00                                  	mov    eax,0x0
   140002e16:	eb 3f                                           	jmp    140002e57 <__mingw_enum_import_library_names+0xdb>
   140002e18:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002e1c:	8b 40 04                                        	mov    eax,DWORD PTR [rax+0x4]
   140002e1f:	85 c0                                           	test   eax,eax
   140002e21:	75 0b                                           	jne    140002e2e <__mingw_enum_import_library_names+0xb2>
   140002e23:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002e27:	8b 40 0c                                        	mov    eax,DWORD PTR [rax+0xc]
   140002e2a:	85 c0                                           	test   eax,eax
   140002e2c:	74 23                                           	je     140002e51 <__mingw_enum_import_library_names+0xd5>
   140002e2e:	83 7d 10 00                                     	cmp    DWORD PTR [rbp+0x10],0x0
   140002e32:	7f 12                                           	jg     140002e46 <__mingw_enum_import_library_names+0xca>
   140002e34:	48 8b 45 f8                                     	mov    rax,QWORD PTR [rbp-0x8]
   140002e38:	8b 40 0c                                        	mov    eax,DWORD PTR [rax+0xc]
   140002e3b:	89 c2                                           	mov    edx,eax
   140002e3d:	48 8b 45 f0                                     	mov    rax,QWORD PTR [rbp-0x10]
   140002e41:	48 01 d0                                        	add    rax,rdx
   140002e44:	eb 11                                           	jmp    140002e57 <__mingw_enum_import_library_names+0xdb>
   140002e46:	83 6d 10 01                                     	sub    DWORD PTR [rbp+0x10],0x1
   140002e4a:	48 83 45 f8 14                                  	add    QWORD PTR [rbp-0x8],0x14
   140002e4f:	eb c7                                           	jmp    140002e18 <__mingw_enum_import_library_names+0x9c>
   140002e51:	90                                              	nop
   140002e52:	b8 00 00 00 00                                  	mov    eax,0x0
   140002e57:	48 83 c4 50                                     	add    rsp,0x50
   140002e5b:	5d                                              	pop    rbp
   140002e5c:	c3                                              	ret
   140002e5d:	90                                              	nop
   140002e5e:	90                                              	nop
   140002e5f:	90                                              	nop

0000000140002e60 <___chkstk_ms>:
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:117
   140002e60:	51                                              	push   rcx
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:119
   140002e61:	50                                              	push   rax
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:121
   140002e62:	48 3d 00 10 00 00                               	cmp    rax,0x1000
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:122
   140002e68:	48 8d 4c 24 18                                  	lea    rcx,[rsp+0x18]
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:123
   140002e6d:	72 19                                           	jb     140002e88 <___chkstk_ms+0x28>
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:125
   140002e6f:	48 81 e9 00 10 00 00                            	sub    rcx,0x1000
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:126
   140002e76:	48 83 09 00                                     	or     QWORD PTR [rcx],0x0
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:127
   140002e7a:	48 2d 00 10 00 00                               	sub    rax,0x1000
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:128
   140002e80:	48 3d 00 10 00 00                               	cmp    rax,0x1000
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:129
   140002e86:	77 e7                                           	ja     140002e6f <___chkstk_ms+0xf>
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:131
   140002e88:	48 29 c1                                        	sub    rcx,rax
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:132
   140002e8b:	48 83 09 00                                     	or     QWORD PTR [rcx],0x0
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:134
   140002e8f:	58                                              	pop    rax
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:136
   140002e90:	59                                              	pop    rcx
R:\winlibs64ucrt_stage\gcc-12.2.0\build_mingw\x86_64-w64-mingw32\libgcc/../../../libgcc/config/i386/cygwin.S:138
   140002e91:	c3                                              	ret
   140002e92:	90                                              	nop
   140002e93:	90                                              	nop
   140002e94:	90                                              	nop
   140002e95:	90                                              	nop
   140002e96:	90                                              	nop
   140002e97:	90                                              	nop
   140002e98:	90                                              	nop
   140002e99:	90                                              	nop
   140002e9a:	90                                              	nop
   140002e9b:	90                                              	nop
   140002e9c:	90                                              	nop
   140002e9d:	90                                              	nop
   140002e9e:	90                                              	nop
   140002e9f:	90                                              	nop

0000000140002ea0 <vfprintf>:
   140002ea0:	55                                              	push   rbp
   140002ea1:	48 89 e5                                        	mov    rbp,rsp
   140002ea4:	48 83 ec 30                                     	sub    rsp,0x30
   140002ea8:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140002eac:	48 89 55 18                                     	mov    QWORD PTR [rbp+0x18],rdx
   140002eb0:	4c 89 45 20                                     	mov    QWORD PTR [rbp+0x20],r8
   140002eb4:	48 8b 4d 18                                     	mov    rcx,QWORD PTR [rbp+0x18]
   140002eb8:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140002ebc:	48 8b 55 20                                     	mov    rdx,QWORD PTR [rbp+0x20]
   140002ec0:	48 89 54 24 20                                  	mov    QWORD PTR [rsp+0x20],rdx
   140002ec5:	41 b9 00 00 00 00                               	mov    r9d,0x0
   140002ecb:	49 89 c8                                        	mov    r8,rcx
   140002ece:	48 89 c2                                        	mov    rdx,rax
   140002ed1:	b9 00 00 00 00                                  	mov    ecx,0x0
   140002ed6:	e8 25 03 00 00                                  	call   140003200 <__stdio_common_vfprintf>
   140002edb:	48 83 c4 30                                     	add    rsp,0x30
   140002edf:	5d                                              	pop    rbp
   140002ee0:	c3                                              	ret
   140002ee1:	90                                              	nop
   140002ee2:	90                                              	nop
   140002ee3:	90                                              	nop
   140002ee4:	90                                              	nop
   140002ee5:	90                                              	nop
   140002ee6:	90                                              	nop
   140002ee7:	90                                              	nop
   140002ee8:	90                                              	nop
   140002ee9:	90                                              	nop
   140002eea:	90                                              	nop
   140002eeb:	90                                              	nop
   140002eec:	90                                              	nop
   140002eed:	90                                              	nop
   140002eee:	90                                              	nop
   140002eef:	90                                              	nop

0000000140002ef0 <fprintf>:
   140002ef0:	55                                              	push   rbp
   140002ef1:	48 89 e5                                        	mov    rbp,rsp
   140002ef4:	48 83 ec 40                                     	sub    rsp,0x40
   140002ef8:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140002efc:	48 89 55 18                                     	mov    QWORD PTR [rbp+0x18],rdx
   140002f00:	4c 89 45 20                                     	mov    QWORD PTR [rbp+0x20],r8
   140002f04:	4c 89 4d 28                                     	mov    QWORD PTR [rbp+0x28],r9
   140002f08:	48 8d 45 20                                     	lea    rax,[rbp+0x20]
   140002f0c:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   140002f10:	48 8b 55 f0                                     	mov    rdx,QWORD PTR [rbp-0x10]
   140002f14:	48 8b 4d 18                                     	mov    rcx,QWORD PTR [rbp+0x18]
   140002f18:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140002f1c:	48 89 54 24 20                                  	mov    QWORD PTR [rsp+0x20],rdx
   140002f21:	41 b9 00 00 00 00                               	mov    r9d,0x0
   140002f27:	49 89 c8                                        	mov    r8,rcx
   140002f2a:	48 89 c2                                        	mov    rdx,rax
   140002f2d:	b9 00 00 00 00                                  	mov    ecx,0x0
   140002f32:	e8 c9 02 00 00                                  	call   140003200 <__stdio_common_vfprintf>
   140002f37:	89 45 fc                                        	mov    DWORD PTR [rbp-0x4],eax
   140002f3a:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   140002f3d:	48 83 c4 40                                     	add    rsp,0x40
   140002f41:	5d                                              	pop    rbp
   140002f42:	c3                                              	ret
   140002f43:	90                                              	nop
   140002f44:	90                                              	nop
   140002f45:	90                                              	nop
   140002f46:	90                                              	nop
   140002f47:	90                                              	nop
   140002f48:	90                                              	nop
   140002f49:	90                                              	nop
   140002f4a:	90                                              	nop
   140002f4b:	90                                              	nop
   140002f4c:	90                                              	nop
   140002f4d:	90                                              	nop
   140002f4e:	90                                              	nop
   140002f4f:	90                                              	nop

0000000140002f50 <__getmainargs>:
   140002f50:	55                                              	push   rbp
   140002f51:	48 89 e5                                        	mov    rbp,rsp
   140002f54:	48 83 ec 20                                     	sub    rsp,0x20
   140002f58:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140002f5c:	48 89 55 18                                     	mov    QWORD PTR [rbp+0x18],rdx
   140002f60:	4c 89 45 20                                     	mov    QWORD PTR [rbp+0x20],r8
   140002f64:	44 89 4d 28                                     	mov    DWORD PTR [rbp+0x28],r9d
   140002f68:	e8 db 02 00 00                                  	call   140003248 <_initialize_narrow_environment>
   140002f6d:	83 7d 28 00                                     	cmp    DWORD PTR [rbp+0x28],0x0
   140002f71:	74 07                                           	je     140002f7a <__getmainargs+0x2a>
   140002f73:	b8 02 00 00 00                                  	mov    eax,0x2
   140002f78:	eb 05                                           	jmp    140002f7f <__getmainargs+0x2f>
   140002f7a:	b8 01 00 00 00                                  	mov    eax,0x1
   140002f7f:	89 c1                                           	mov    ecx,eax
   140002f81:	e8 a2 02 00 00                                  	call   140003228 <_configure_narrow_argv>
   140002f86:	e8 2d 02 00 00                                  	call   1400031b8 <__p___argc>
   140002f8b:	8b 10                                           	mov    edx,DWORD PTR [rax]
   140002f8d:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140002f91:	89 10                                           	mov    DWORD PTR [rax],edx
   140002f93:	e8 28 02 00 00                                  	call   1400031c0 <__p___argv>
   140002f98:	48 8b 10                                        	mov    rdx,QWORD PTR [rax]
   140002f9b:	48 8b 45 18                                     	mov    rax,QWORD PTR [rbp+0x18]
   140002f9f:	48 89 10                                        	mov    QWORD PTR [rax],rdx
   140002fa2:	e8 39 02 00 00                                  	call   1400031e0 <__p__environ>
   140002fa7:	48 8b 10                                        	mov    rdx,QWORD PTR [rax]
   140002faa:	48 8b 45 20                                     	mov    rax,QWORD PTR [rbp+0x20]
   140002fae:	48 89 10                                        	mov    QWORD PTR [rax],rdx
   140002fb1:	48 83 7d 30 00                                  	cmp    QWORD PTR [rbp+0x30],0x0
   140002fb6:	74 0d                                           	je     140002fc5 <__getmainargs+0x75>
   140002fb8:	48 8b 45 30                                     	mov    rax,QWORD PTR [rbp+0x30]
   140002fbc:	8b 00                                           	mov    eax,DWORD PTR [rax]
   140002fbe:	89 c1                                           	mov    ecx,eax
   140002fc0:	e8 ab 02 00 00                                  	call   140003270 <_set_new_mode>
   140002fc5:	b8 00 00 00 00                                  	mov    eax,0x0
   140002fca:	48 83 c4 20                                     	add    rsp,0x20
   140002fce:	5d                                              	pop    rbp
   140002fcf:	c3                                              	ret

0000000140002fd0 <__wgetmainargs>:
   140002fd0:	55                                              	push   rbp
   140002fd1:	48 89 e5                                        	mov    rbp,rsp
   140002fd4:	48 83 ec 20                                     	sub    rsp,0x20
   140002fd8:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140002fdc:	48 89 55 18                                     	mov    QWORD PTR [rbp+0x18],rdx
   140002fe0:	4c 89 45 20                                     	mov    QWORD PTR [rbp+0x20],r8
   140002fe4:	44 89 4d 28                                     	mov    DWORD PTR [rbp+0x28],r9d
   140002fe8:	e8 63 02 00 00                                  	call   140003250 <_initialize_wide_environment>
   140002fed:	83 7d 28 00                                     	cmp    DWORD PTR [rbp+0x28],0x0
   140002ff1:	74 07                                           	je     140002ffa <__wgetmainargs+0x2a>
   140002ff3:	b8 02 00 00 00                                  	mov    eax,0x2
   140002ff8:	eb 05                                           	jmp    140002fff <__wgetmainargs+0x2f>
   140002ffa:	b8 01 00 00 00                                  	mov    eax,0x1
   140002fff:	89 c1                                           	mov    ecx,eax
   140003001:	e8 2a 02 00 00                                  	call   140003230 <_configure_wide_argv>
   140003006:	e8 ad 01 00 00                                  	call   1400031b8 <__p___argc>
   14000300b:	8b 10                                           	mov    edx,DWORD PTR [rax]
   14000300d:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140003011:	89 10                                           	mov    DWORD PTR [rax],edx
   140003013:	e8 b0 01 00 00                                  	call   1400031c8 <__p___wargv>
   140003018:	48 8b 10                                        	mov    rdx,QWORD PTR [rax]
   14000301b:	48 8b 45 18                                     	mov    rax,QWORD PTR [rbp+0x18]
   14000301f:	48 89 10                                        	mov    QWORD PTR [rax],rdx
   140003022:	e8 c9 01 00 00                                  	call   1400031f0 <__p__wenviron>
   140003027:	48 8b 10                                        	mov    rdx,QWORD PTR [rax]
   14000302a:	48 8b 45 20                                     	mov    rax,QWORD PTR [rbp+0x20]
   14000302e:	48 89 10                                        	mov    QWORD PTR [rax],rdx
   140003031:	48 83 7d 30 00                                  	cmp    QWORD PTR [rbp+0x30],0x0
   140003036:	74 0d                                           	je     140003045 <__wgetmainargs+0x75>
   140003038:	48 8b 45 30                                     	mov    rax,QWORD PTR [rbp+0x30]
   14000303c:	8b 00                                           	mov    eax,DWORD PTR [rax]
   14000303e:	89 c1                                           	mov    ecx,eax
   140003040:	e8 2b 02 00 00                                  	call   140003270 <_set_new_mode>
   140003045:	b8 00 00 00 00                                  	mov    eax,0x0
   14000304a:	48 83 c4 20                                     	add    rsp,0x20
   14000304e:	5d                                              	pop    rbp
   14000304f:	c3                                              	ret

0000000140003050 <_onexit>:
   140003050:	55                                              	push   rbp
   140003051:	48 89 e5                                        	mov    rbp,rsp
   140003054:	48 83 ec 20                                     	sub    rsp,0x20
   140003058:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   14000305c:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140003060:	48 89 c1                                        	mov    rcx,rax
   140003063:	e8 d8 01 00 00                                  	call   140003240 <_crt_atexit>
   140003068:	85 c0                                           	test   eax,eax
   14000306a:	75 06                                           	jne    140003072 <_onexit+0x22>
   14000306c:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   140003070:	eb 05                                           	jmp    140003077 <_onexit+0x27>
   140003072:	b8 00 00 00 00                                  	mov    eax,0x0
   140003077:	48 83 c4 20                                     	add    rsp,0x20
   14000307b:	5d                                              	pop    rbp
   14000307c:	c3                                              	ret

000000014000307d <at_quick_exit>:
   14000307d:	55                                              	push   rbp
   14000307e:	48 89 e5                                        	mov    rbp,rsp
   140003081:	48 83 ec 20                                     	sub    rsp,0x20
   140003085:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   140003089:	48 8b 05 e0 23 00 00                            	mov    rax,QWORD PTR [rip+0x23e0]        # 140005470 <.refptr.__mingw_module_is_dll>
   140003090:	0f b6 00                                        	movzx  eax,BYTE PTR [rax]
   140003093:	84 c0                                           	test   al,al
   140003095:	74 07                                           	je     14000309e <at_quick_exit+0x21>
   140003097:	b8 00 00 00 00                                  	mov    eax,0x0
   14000309c:	eb 0c                                           	jmp    1400030aa <at_quick_exit+0x2d>
   14000309e:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   1400030a2:	48 89 c1                                        	mov    rcx,rax
   1400030a5:	e8 8e 01 00 00                                  	call   140003238 <_crt_at_quick_exit>
   1400030aa:	48 83 c4 20                                     	add    rsp,0x20
   1400030ae:	5d                                              	pop    rbp
   1400030af:	c3                                              	ret

00000001400030b0 <_amsg_exit>:
   1400030b0:	55                                              	push   rbp
   1400030b1:	48 89 e5                                        	mov    rbp,rsp
   1400030b4:	48 83 ec 20                                     	sub    rsp,0x20
   1400030b8:	89 4d 10                                        	mov    DWORD PTR [rbp+0x10],ecx
   1400030bb:	b9 02 00 00 00                                  	mov    ecx,0x2
   1400030c0:	e8 e3 00 00 00                                  	call   1400031a8 <__acrt_iob_func>
   1400030c5:	48 89 c1                                        	mov    rcx,rax
   1400030c8:	8b 45 10                                        	mov    eax,DWORD PTR [rbp+0x10]
   1400030cb:	41 89 c0                                        	mov    r8d,eax
   1400030ce:	48 8d 05 9b 22 00 00                            	lea    rax,[rip+0x229b]        # 140005370 <.rdata>
   1400030d5:	48 89 c2                                        	mov    rdx,rax
   1400030d8:	e8 13 fe ff ff                                  	call   140002ef0 <fprintf>
   1400030dd:	90                                              	nop
   1400030de:	48 83 c4 20                                     	add    rsp,0x20
   1400030e2:	5d                                              	pop    rbp
   1400030e3:	c3                                              	ret

00000001400030e4 <_get_output_format>:
   1400030e4:	55                                              	push   rbp
   1400030e5:	48 89 e5                                        	mov    rbp,rsp
   1400030e8:	b8 00 00 00 00                                  	mov    eax,0x0
   1400030ed:	5d                                              	pop    rbp
   1400030ee:	c3                                              	ret

00000001400030ef <_tzset>:
   1400030ef:	55                                              	push   rbp
   1400030f0:	48 89 e5                                        	mov    rbp,rsp
   1400030f3:	48 83 ec 20                                     	sub    rsp,0x20
   1400030f7:	48 8b 05 22 23 00 00                            	mov    rax,QWORD PTR [rip+0x2322]        # 140005420 <.refptr.__imp__tzset>
   1400030fe:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
   140003101:	ff d0                                           	call   rax
   140003103:	e8 10 01 00 00                                  	call   140003218 <__tzname>
   140003108:	48 89 05 d9 0f 00 00                            	mov    QWORD PTR [rip+0xfd9],rax        # 1400040e8 <__imp_tzname>
   14000310f:	e8 fc 00 00 00                                  	call   140003210 <__timezone>
   140003114:	48 89 05 d5 0f 00 00                            	mov    QWORD PTR [rip+0xfd5],rax        # 1400040f0 <__imp_timezone>
   14000311b:	e8 90 00 00 00                                  	call   1400031b0 <__daylight>
   140003120:	48 89 05 d1 0f 00 00                            	mov    QWORD PTR [rip+0xfd1],rax        # 1400040f8 <__imp_daylight>
   140003127:	90                                              	nop
   140003128:	48 83 c4 20                                     	add    rsp,0x20
   14000312c:	5d                                              	pop    rbp
   14000312d:	c3                                              	ret

000000014000312e <tzset>:
   14000312e:	55                                              	push   rbp
   14000312f:	48 89 e5                                        	mov    rbp,rsp
   140003132:	48 83 ec 20                                     	sub    rsp,0x20
   140003136:	e8 b4 ff ff ff                                  	call   1400030ef <_tzset>
   14000313b:	90                                              	nop
   14000313c:	48 83 c4 20                                     	add    rsp,0x20
   140003140:	5d                                              	pop    rbp
   140003141:	c3                                              	ret

0000000140003142 <__ms_fwprintf>:
   140003142:	55                                              	push   rbp
   140003143:	48 89 e5                                        	mov    rbp,rsp
   140003146:	48 83 ec 40                                     	sub    rsp,0x40
   14000314a:	48 89 4d 10                                     	mov    QWORD PTR [rbp+0x10],rcx
   14000314e:	48 89 55 18                                     	mov    QWORD PTR [rbp+0x18],rdx
   140003152:	4c 89 45 20                                     	mov    QWORD PTR [rbp+0x20],r8
   140003156:	4c 89 4d 28                                     	mov    QWORD PTR [rbp+0x28],r9
   14000315a:	48 8d 45 20                                     	lea    rax,[rbp+0x20]
   14000315e:	48 89 45 f0                                     	mov    QWORD PTR [rbp-0x10],rax
   140003162:	48 8b 55 f0                                     	mov    rdx,QWORD PTR [rbp-0x10]
   140003166:	48 8b 4d 18                                     	mov    rcx,QWORD PTR [rbp+0x18]
   14000316a:	48 8b 45 10                                     	mov    rax,QWORD PTR [rbp+0x10]
   14000316e:	48 89 54 24 20                                  	mov    QWORD PTR [rsp+0x20],rdx
   140003173:	41 b9 00 00 00 00                               	mov    r9d,0x0
   140003179:	49 89 c8                                        	mov    r8,rcx
   14000317c:	48 89 c2                                        	mov    rdx,rax
   14000317f:	b9 04 00 00 00                                  	mov    ecx,0x4
   140003184:	e8 7f 00 00 00                                  	call   140003208 <__stdio_common_vfwprintf>
   140003189:	89 45 fc                                        	mov    DWORD PTR [rbp-0x4],eax
   14000318c:	8b 45 fc                                        	mov    eax,DWORD PTR [rbp-0x4]
   14000318f:	48 83 c4 40                                     	add    rsp,0x40
   140003193:	5d                                              	pop    rbp
   140003194:	c3                                              	ret
   140003195:	90                                              	nop
   140003196:	90                                              	nop
   140003197:	90                                              	nop
   140003198:	90                                              	nop
   140003199:	90                                              	nop
   14000319a:	90                                              	nop
   14000319b:	90                                              	nop
   14000319c:	90                                              	nop
   14000319d:	90                                              	nop
   14000319e:	90                                              	nop
   14000319f:	90                                              	nop

00000001400031a0 <__C_specific_handler>:
   1400031a0:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a280 <__imp___C_specific_handler>
   1400031a6:	90                                              	nop
   1400031a7:	90                                              	nop

00000001400031a8 <__acrt_iob_func>:
   1400031a8:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a288 <__imp___acrt_iob_func>
   1400031ae:	90                                              	nop
   1400031af:	90                                              	nop

00000001400031b0 <__daylight>:
   1400031b0:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a290 <__imp___daylight>
   1400031b6:	90                                              	nop
   1400031b7:	90                                              	nop

00000001400031b8 <__p___argc>:
   1400031b8:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a298 <__imp___p___argc>
   1400031be:	90                                              	nop
   1400031bf:	90                                              	nop

00000001400031c0 <__p___argv>:
   1400031c0:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a2a0 <__imp___p___argv>
   1400031c6:	90                                              	nop
   1400031c7:	90                                              	nop

00000001400031c8 <__p___wargv>:
   1400031c8:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a2a8 <__imp___p___wargv>
   1400031ce:	90                                              	nop
   1400031cf:	90                                              	nop

00000001400031d0 <__p__acmdln>:
   1400031d0:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a2b0 <__imp___p__acmdln>
   1400031d6:	90                                              	nop
   1400031d7:	90                                              	nop

00000001400031d8 <__p__commode>:
   1400031d8:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a2b8 <__imp___p__commode>
   1400031de:	90                                              	nop
   1400031df:	90                                              	nop

00000001400031e0 <__p__environ>:
   1400031e0:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a2c0 <__imp___p__environ>
   1400031e6:	90                                              	nop
   1400031e7:	90                                              	nop

00000001400031e8 <__p__fmode>:
   1400031e8:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a2c8 <__imp___p__fmode>
   1400031ee:	90                                              	nop
   1400031ef:	90                                              	nop

00000001400031f0 <__p__wenviron>:
   1400031f0:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a2d0 <__imp___p__wenviron>
   1400031f6:	90                                              	nop
   1400031f7:	90                                              	nop

00000001400031f8 <__setusermatherr>:
   1400031f8:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a2d8 <__imp___setusermatherr>
   1400031fe:	90                                              	nop
   1400031ff:	90                                              	nop

0000000140003200 <__stdio_common_vfprintf>:
   140003200:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a2e0 <__imp___stdio_common_vfprintf>
   140003206:	90                                              	nop
   140003207:	90                                              	nop

0000000140003208 <__stdio_common_vfwprintf>:
   140003208:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a2e8 <__imp___stdio_common_vfwprintf>
   14000320e:	90                                              	nop
   14000320f:	90                                              	nop

0000000140003210 <__timezone>:
   140003210:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a2f0 <__imp___timezone>
   140003216:	90                                              	nop
   140003217:	90                                              	nop

0000000140003218 <__tzname>:
   140003218:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a2f8 <__imp___tzname>
   14000321e:	90                                              	nop
   14000321f:	90                                              	nop

0000000140003220 <_cexit>:
   140003220:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a300 <__imp__cexit>
   140003226:	90                                              	nop
   140003227:	90                                              	nop

0000000140003228 <_configure_narrow_argv>:
   140003228:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a308 <__imp__configure_narrow_argv>
   14000322e:	90                                              	nop
   14000322f:	90                                              	nop

0000000140003230 <_configure_wide_argv>:
   140003230:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a310 <__imp__configure_wide_argv>
   140003236:	90                                              	nop
   140003237:	90                                              	nop

0000000140003238 <_crt_at_quick_exit>:
   140003238:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a318 <__imp__crt_at_quick_exit>
   14000323e:	90                                              	nop
   14000323f:	90                                              	nop

0000000140003240 <_crt_atexit>:
   140003240:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a320 <__imp__crt_atexit>
   140003246:	90                                              	nop
   140003247:	90                                              	nop

0000000140003248 <_initialize_narrow_environment>:
   140003248:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a328 <__imp__initialize_narrow_environment>
   14000324e:	90                                              	nop
   14000324f:	90                                              	nop

0000000140003250 <_initialize_wide_environment>:
   140003250:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a330 <__imp__initialize_wide_environment>
   140003256:	90                                              	nop
   140003257:	90                                              	nop

0000000140003258 <_initterm>:
   140003258:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a338 <__imp__initterm>
   14000325e:	90                                              	nop
   14000325f:	90                                              	nop

0000000140003260 <__set_app_type>:
   140003260:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a340 <__imp___set_app_type>
   140003266:	90                                              	nop
   140003267:	90                                              	nop

0000000140003268 <_set_invalid_parameter_handler>:
   140003268:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a348 <__imp__set_invalid_parameter_handler>
   14000326e:	90                                              	nop
   14000326f:	90                                              	nop

0000000140003270 <_set_new_mode>:
   140003270:	ff 25 da 70 00 00                               	jmp    QWORD PTR [rip+0x70da]        # 14000a350 <__imp__set_new_mode>
   140003276:	90                                              	nop
   140003277:	90                                              	nop

0000000140003278 <abort>:
   140003278:	ff 25 e2 70 00 00                               	jmp    QWORD PTR [rip+0x70e2]        # 14000a360 <__imp_abort>
   14000327e:	90                                              	nop
   14000327f:	90                                              	nop

0000000140003280 <calloc>:
   140003280:	ff 25 e2 70 00 00                               	jmp    QWORD PTR [rip+0x70e2]        # 14000a368 <__imp_calloc>
   140003286:	90                                              	nop
   140003287:	90                                              	nop

0000000140003288 <exit>:
   140003288:	ff 25 e2 70 00 00                               	jmp    QWORD PTR [rip+0x70e2]        # 14000a370 <__imp_exit>
   14000328e:	90                                              	nop
   14000328f:	90                                              	nop

0000000140003290 <free>:
   140003290:	ff 25 e2 70 00 00                               	jmp    QWORD PTR [rip+0x70e2]        # 14000a378 <__imp_free>
   140003296:	90                                              	nop
   140003297:	90                                              	nop

0000000140003298 <fwrite>:
   140003298:	ff 25 e2 70 00 00                               	jmp    QWORD PTR [rip+0x70e2]        # 14000a380 <__imp_fwrite>
   14000329e:	90                                              	nop
   14000329f:	90                                              	nop

00000001400032a0 <malloc>:
   1400032a0:	ff 25 e2 70 00 00                               	jmp    QWORD PTR [rip+0x70e2]        # 14000a388 <__imp_malloc>
   1400032a6:	90                                              	nop
   1400032a7:	90                                              	nop

00000001400032a8 <memcpy>:
   1400032a8:	ff 25 e2 70 00 00                               	jmp    QWORD PTR [rip+0x70e2]        # 14000a390 <__imp_memcpy>
   1400032ae:	90                                              	nop
   1400032af:	90                                              	nop

00000001400032b0 <memset>:
   1400032b0:	ff 25 e2 70 00 00                               	jmp    QWORD PTR [rip+0x70e2]        # 14000a398 <__imp_memset>
   1400032b6:	90                                              	nop
   1400032b7:	90                                              	nop

00000001400032b8 <signal>:
   1400032b8:	ff 25 e2 70 00 00                               	jmp    QWORD PTR [rip+0x70e2]        # 14000a3a0 <__imp_signal>
   1400032be:	90                                              	nop
   1400032bf:	90                                              	nop

00000001400032c0 <strlen>:
   1400032c0:	ff 25 e2 70 00 00                               	jmp    QWORD PTR [rip+0x70e2]        # 14000a3a8 <__imp_strlen>
   1400032c6:	90                                              	nop
   1400032c7:	90                                              	nop

00000001400032c8 <strncmp>:
   1400032c8:	ff 25 e2 70 00 00                               	jmp    QWORD PTR [rip+0x70e2]        # 14000a3b0 <__imp_strncmp>
   1400032ce:	90                                              	nop
   1400032cf:	90                                              	nop

00000001400032d0 <VirtualQuery>:
   1400032d0:	ff 25 9a 6f 00 00                               	jmp    QWORD PTR [rip+0x6f9a]        # 14000a270 <__imp_VirtualQuery>
   1400032d6:	90                                              	nop
   1400032d7:	90                                              	nop

00000001400032d8 <VirtualProtect>:
   1400032d8:	ff 25 8a 6f 00 00                               	jmp    QWORD PTR [rip+0x6f8a]        # 14000a268 <__imp_VirtualProtect>
   1400032de:	90                                              	nop
   1400032df:	90                                              	nop

00000001400032e0 <TlsGetValue>:
   1400032e0:	ff 25 7a 6f 00 00                               	jmp    QWORD PTR [rip+0x6f7a]        # 14000a260 <__imp_TlsGetValue>
   1400032e6:	90                                              	nop
   1400032e7:	90                                              	nop

00000001400032e8 <Sleep>:
   1400032e8:	ff 25 6a 6f 00 00                               	jmp    QWORD PTR [rip+0x6f6a]        # 14000a258 <__imp_Sleep>
   1400032ee:	90                                              	nop
   1400032ef:	90                                              	nop

00000001400032f0 <SetUnhandledExceptionFilter>:
   1400032f0:	ff 25 5a 6f 00 00                               	jmp    QWORD PTR [rip+0x6f5a]        # 14000a250 <__imp_SetUnhandledExceptionFilter>
   1400032f6:	90                                              	nop
   1400032f7:	90                                              	nop

00000001400032f8 <LoadLibraryA>:
   1400032f8:	ff 25 4a 6f 00 00                               	jmp    QWORD PTR [rip+0x6f4a]        # 14000a248 <__imp_LoadLibraryA>
   1400032fe:	90                                              	nop
   1400032ff:	90                                              	nop

0000000140003300 <LeaveCriticalSection>:
   140003300:	ff 25 3a 6f 00 00                               	jmp    QWORD PTR [rip+0x6f3a]        # 14000a240 <__imp_LeaveCriticalSection>
   140003306:	90                                              	nop
   140003307:	90                                              	nop

0000000140003308 <InitializeCriticalSection>:
   140003308:	ff 25 2a 6f 00 00                               	jmp    QWORD PTR [rip+0x6f2a]        # 14000a238 <__imp_InitializeCriticalSection>
   14000330e:	90                                              	nop
   14000330f:	90                                              	nop

0000000140003310 <GetStartupInfoA>:
   140003310:	ff 25 1a 6f 00 00                               	jmp    QWORD PTR [rip+0x6f1a]        # 14000a230 <__imp_GetStartupInfoA>
   140003316:	90                                              	nop
   140003317:	90                                              	nop

0000000140003318 <GetProcAddress>:
   140003318:	ff 25 0a 6f 00 00                               	jmp    QWORD PTR [rip+0x6f0a]        # 14000a228 <__imp_GetProcAddress>
   14000331e:	90                                              	nop
   14000331f:	90                                              	nop

0000000140003320 <GetModuleHandleA>:
   140003320:	ff 25 fa 6e 00 00                               	jmp    QWORD PTR [rip+0x6efa]        # 14000a220 <__imp_GetModuleHandleA>
   140003326:	90                                              	nop
   140003327:	90                                              	nop

0000000140003328 <GetLastError>:
   140003328:	ff 25 ea 6e 00 00                               	jmp    QWORD PTR [rip+0x6eea]        # 14000a218 <__imp_GetLastError>
   14000332e:	90                                              	nop
   14000332f:	90                                              	nop

0000000140003330 <FreeLibrary>:
   140003330:	ff 25 da 6e 00 00                               	jmp    QWORD PTR [rip+0x6eda]        # 14000a210 <__imp_FreeLibrary>
   140003336:	90                                              	nop
   140003337:	90                                              	nop

0000000140003338 <EnterCriticalSection>:
   140003338:	ff 25 ca 6e 00 00                               	jmp    QWORD PTR [rip+0x6eca]        # 14000a208 <__imp_EnterCriticalSection>
   14000333e:	90                                              	nop
   14000333f:	90                                              	nop

0000000140003340 <DeleteCriticalSection>:
   140003340:	ff 25 ba 6e 00 00                               	jmp    QWORD PTR [rip+0x6eba]        # 14000a200 <__IAT_start__>
   140003346:	90                                              	nop
   140003347:	90                                              	nop
   140003348:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]

0000000140003350 <register_frame_ctor>:
   140003350:	e9 8b e3 ff ff                                  	jmp    1400016e0 <__gcc_register_frame>
   140003355:	90                                              	nop
   140003356:	90                                              	nop
   140003357:	90                                              	nop
   140003358:	90                                              	nop
   140003359:	90                                              	nop
   14000335a:	90                                              	nop
   14000335b:	90                                              	nop
   14000335c:	90                                              	nop
   14000335d:	90                                              	nop
   14000335e:	90                                              	nop
   14000335f:	90                                              	nop

0000000140003360 <__CTOR_LIST__>:
   140003360:	ff                                              	(bad)
   140003361:	ff                                              	(bad)
   140003362:	ff                                              	(bad)
   140003363:	ff                                              	(bad)
   140003364:	ff                                              	(bad)
   140003365:	ff                                              	(bad)
   140003366:	ff                                              	(bad)
   140003367:	ff                                              	.byte 0xff

0000000140003368 <.ctors.65535>:
   140003368:	50                                              	push   rax
   140003369:	33 00                                           	xor    eax,DWORD PTR [rax]
   14000336b:	40 01 00                                        	rex add DWORD PTR [rax],eax
	...

0000000140003378 <__DTOR_LIST__>:
   140003378:	ff                                              	(bad)
   140003379:	ff                                              	(bad)
   14000337a:	ff                                              	(bad)
   14000337b:	ff                                              	(bad)
   14000337c:	ff                                              	(bad)
   14000337d:	ff                                              	(bad)
   14000337e:	ff                                              	(bad)
   14000337f:	ff 00                                           	inc    DWORD PTR [rax]
   140003381:	00 00                                           	add    BYTE PTR [rax],al
   140003383:	00 00                                           	add    BYTE PTR [rax],al
   140003385:	00 00                                           	add    BYTE PTR [rax],al
	...
