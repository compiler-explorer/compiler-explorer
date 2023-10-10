
/tmp/compiler-explorer-compiler202095-9137-1kkxb4c.yd7w/output.s:     file format elf64-x86-64


Disassembly of section .init:

0000000000402000 <_init>:
_init():
  402000:	f3 0f 1e fa                                     	endbr64
  402004:	48 83 ec 08                                     	sub    rsp,0x8
  402008:	48 8b 05 d1 7f 01 00                            	mov    rax,QWORD PTR [rip+0x17fd1]        # 419fe0 <__gmon_start__>
  40200f:	48 85 c0                                        	test   rax,rax
  402012:	74 02                                           	je     402016 <_init+0x16>
  402014:	ff d0                                           	call   rax
  402016:	48 83 c4 08                                     	add    rsp,0x8
  40201a:	c3                                              	ret

Disassembly of section .plt:

0000000000402020 <.plt>:
  402020:	ff 35 e2 7f 01 00                               	push   QWORD PTR [rip+0x17fe2]        # 41a008 <_GLOBAL_OFFSET_TABLE_+0x8>
  402026:	ff 25 e4 7f 01 00                               	jmp    QWORD PTR [rip+0x17fe4]        # 41a010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40202c:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

0000000000402030 <_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_@plt>:
  402030:	ff 25 e2 7f 01 00                               	jmp    QWORD PTR [rip+0x17fe2]        # 41a018 <_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_@GLIBCXX_3.4>
  402036:	68 00 00 00 00                                  	push   0x0
  40203b:	e9 e0 ff ff ff                                  	jmp    402020 <.plt>

0000000000402040 <_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7compareEPKc@plt>:
  402040:	ff 25 da 7f 01 00                               	jmp    QWORD PTR [rip+0x17fda]        # 41a020 <_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7compareEPKc@GLIBCXX_3.4.21>
  402046:	68 01 00 00 00                                  	push   0x1
  40204b:	e9 d0 ff ff ff                                  	jmp    402020 <.plt>

0000000000402050 <_ZNSt8ios_baseC2Ev@plt>:
  402050:	ff 25 d2 7f 01 00                               	jmp    QWORD PTR [rip+0x17fd2]        # 41a028 <_ZNSt8ios_baseC2Ev@GLIBCXX_3.4>
  402056:	68 02 00 00 00                                  	push   0x2
  40205b:	e9 c0 ff ff ff                                  	jmp    402020 <.plt>

0000000000402060 <_ZSt19__throw_regex_errorNSt15regex_constants10error_typeE@plt>:
  402060:	ff 25 ca 7f 01 00                               	jmp    QWORD PTR [rip+0x17fca]        # 41a030 <_ZSt19__throw_regex_errorNSt15regex_constants10error_typeE@GLIBCXX_3.4.15>
  402066:	68 03 00 00 00                                  	push   0x3
  40206b:	e9 b0 ff ff ff                                  	jmp    402020 <.plt>

0000000000402070 <_ZNSt8ios_baseD2Ev@plt>:
  402070:	ff 25 c2 7f 01 00                               	jmp    QWORD PTR [rip+0x17fc2]        # 41a038 <_ZNSt8ios_baseD2Ev@GLIBCXX_3.4>
  402076:	68 04 00 00 00                                  	push   0x4
  40207b:	e9 a0 ff ff ff                                  	jmp    402020 <.plt>

0000000000402080 <_ZSt17__throw_bad_allocv@plt>:
  402080:	ff 25 ba 7f 01 00                               	jmp    QWORD PTR [rip+0x17fba]        # 41a040 <_ZSt17__throw_bad_allocv@GLIBCXX_3.4>
  402086:	68 05 00 00 00                                  	push   0x5
  40208b:	e9 90 ff ff ff                                  	jmp    402020 <.plt>

0000000000402090 <strchr@plt>:
  402090:	ff 25 b2 7f 01 00                               	jmp    QWORD PTR [rip+0x17fb2]        # 41a048 <strchr@GLIBC_2.2.5>
  402096:	68 06 00 00 00                                  	push   0x6
  40209b:	e9 80 ff ff ff                                  	jmp    402020 <.plt>

00000000004020a0 <__cxa_begin_catch@plt>:
  4020a0:	ff 25 aa 7f 01 00                               	jmp    QWORD PTR [rip+0x17faa]        # 41a050 <__cxa_begin_catch@CXXABI_1.3>
  4020a6:	68 07 00 00 00                                  	push   0x7
  4020ab:	e9 70 ff ff ff                                  	jmp    402020 <.plt>

00000000004020b0 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@plt>:
  4020b0:	ff 25 a2 7f 01 00                               	jmp    QWORD PTR [rip+0x17fa2]        # 41a058 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@GLIBCXX_3.4>
  4020b6:	68 08 00 00 00                                  	push   0x8
  4020bb:	e9 60 ff ff ff                                  	jmp    402020 <.plt>

00000000004020c0 <memcmp@plt>:
  4020c0:	ff 25 9a 7f 01 00                               	jmp    QWORD PTR [rip+0x17f9a]        # 41a060 <memcmp@GLIBC_2.2.5>
  4020c6:	68 09 00 00 00                                  	push   0x9
  4020cb:	e9 50 ff ff ff                                  	jmp    402020 <.plt>

00000000004020d0 <_ZNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEED1Ev@plt>:
  4020d0:	ff 25 92 7f 01 00                               	jmp    QWORD PTR [rip+0x17f92]        # 41a068 <_ZNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEED1Ev@GLIBCXX_3.4.21>
  4020d6:	68 0a 00 00 00                                  	push   0xa
  4020db:	e9 40 ff ff ff                                  	jmp    402020 <.plt>

00000000004020e0 <__cxa_allocate_exception@plt>:
  4020e0:	ff 25 8a 7f 01 00                               	jmp    QWORD PTR [rip+0x17f8a]        # 41a070 <__cxa_allocate_exception@CXXABI_1.3>
  4020e6:	68 0b 00 00 00                                  	push   0xb
  4020eb:	e9 30 ff ff ff                                  	jmp    402020 <.plt>

00000000004020f0 <_ZNKSt6locale2id5_M_idEv@plt>:
  4020f0:	ff 25 82 7f 01 00                               	jmp    QWORD PTR [rip+0x17f82]        # 41a078 <_ZNKSt6locale2id5_M_idEv@GLIBCXX_3.4>
  4020f6:	68 0c 00 00 00                                  	push   0xc
  4020fb:	e9 20 ff ff ff                                  	jmp    402020 <.plt>

0000000000402100 <_ZSt20__throw_length_errorPKc@plt>:
  402100:	ff 25 7a 7f 01 00                               	jmp    QWORD PTR [rip+0x17f7a]        # 41a080 <_ZSt20__throw_length_errorPKc@GLIBCXX_3.4>
  402106:	68 0d 00 00 00                                  	push   0xd
  40210b:	e9 10 ff ff ff                                  	jmp    402020 <.plt>

0000000000402110 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_@plt>:
  402110:	ff 25 72 7f 01 00                               	jmp    QWORD PTR [rip+0x17f72]        # 41a088 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_@GLIBCXX_3.4.21>
  402116:	68 0e 00 00 00                                  	push   0xe
  40211b:	e9 00 ff ff ff                                  	jmp    402020 <.plt>

0000000000402120 <_ZNSt6localeC1ERKS_@plt>:
  402120:	ff 25 6a 7f 01 00                               	jmp    QWORD PTR [rip+0x17f6a]        # 41a090 <_ZNSt6localeC1ERKS_@GLIBCXX_3.4>
  402126:	68 0f 00 00 00                                  	push   0xf
  40212b:	e9 f0 fe ff ff                                  	jmp    402020 <.plt>

0000000000402130 <__cxa_guard_abort@plt>:
  402130:	ff 25 62 7f 01 00                               	jmp    QWORD PTR [rip+0x17f62]        # 41a098 <__cxa_guard_abort@CXXABI_1.3>
  402136:	68 10 00 00 00                                  	push   0x10
  40213b:	e9 e0 fe ff ff                                  	jmp    402020 <.plt>

0000000000402140 <__cxa_guard_release@plt>:
  402140:	ff 25 5a 7f 01 00                               	jmp    QWORD PTR [rip+0x17f5a]        # 41a0a0 <__cxa_guard_release@CXXABI_1.3>
  402146:	68 11 00 00 00                                  	push   0x11
  40214b:	e9 d0 fe ff ff                                  	jmp    402020 <.plt>

0000000000402150 <_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base@plt>:
  402150:	ff 25 52 7f 01 00                               	jmp    QWORD PTR [rip+0x17f52]        # 41a0a8 <_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base@GLIBCXX_3.4>
  402156:	68 12 00 00 00                                  	push   0x12
  40215b:	e9 c0 fe ff ff                                  	jmp    402020 <.plt>

0000000000402160 <_ZSt19__throw_logic_errorPKc@plt>:
  402160:	ff 25 4a 7f 01 00                               	jmp    QWORD PTR [rip+0x17f4a]        # 41a0b0 <_ZSt19__throw_logic_errorPKc@GLIBCXX_3.4>
  402166:	68 13 00 00 00                                  	push   0x13
  40216b:	e9 b0 fe ff ff                                  	jmp    402020 <.plt>

0000000000402170 <__cxa_free_exception@plt>:
  402170:	ff 25 42 7f 01 00                               	jmp    QWORD PTR [rip+0x17f42]        # 41a0b8 <__cxa_free_exception@CXXABI_1.3>
  402176:	68 14 00 00 00                                  	push   0x14
  40217b:	e9 a0 fe ff ff                                  	jmp    402020 <.plt>

0000000000402180 <memcpy@plt>:
  402180:	ff 25 3a 7f 01 00                               	jmp    QWORD PTR [rip+0x17f3a]        # 41a0c0 <memcpy@GLIBC_2.14>
  402186:	68 15 00 00 00                                  	push   0x15
  40218b:	e9 90 fe ff ff                                  	jmp    402020 <.plt>

0000000000402190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc@plt>:
  402190:	ff 25 32 7f 01 00                               	jmp    QWORD PTR [rip+0x17f32]        # 41a0c8 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc@GLIBCXX_3.4.21>
  402196:	68 16 00 00 00                                  	push   0x16
  40219b:	e9 80 fe ff ff                                  	jmp    402020 <.plt>

00000000004021a0 <_ZSt9use_facetINSt7__cxx117collateIcEEERKT_RKSt6locale@plt>:
  4021a0:	ff 25 2a 7f 01 00                               	jmp    QWORD PTR [rip+0x17f2a]        # 41a0d0 <_ZSt9use_facetINSt7__cxx117collateIcEEERKT_RKSt6locale@GLIBCXX_3.4.21>
  4021a6:	68 17 00 00 00                                  	push   0x17
  4021ab:	e9 70 fe ff ff                                  	jmp    402020 <.plt>

00000000004021b0 <_Znwm@plt>:
  4021b0:	ff 25 22 7f 01 00                               	jmp    QWORD PTR [rip+0x17f22]        # 41a0d8 <_Znwm@GLIBCXX_3.4>
  4021b6:	68 18 00 00 00                                  	push   0x18
  4021bb:	e9 60 fe ff ff                                  	jmp    402020 <.plt>

00000000004021c0 <_ZdlPvm@plt>:
  4021c0:	ff 25 1a 7f 01 00                               	jmp    QWORD PTR [rip+0x17f1a]        # 41a0e0 <_ZdlPvm@CXXABI_1.3.9>
  4021c6:	68 19 00 00 00                                  	push   0x19
  4021cb:	e9 50 fe ff ff                                  	jmp    402020 <.plt>

00000000004021d0 <_ZNSi10_M_extractIlEERSiRT_@plt>:
  4021d0:	ff 25 12 7f 01 00                               	jmp    QWORD PTR [rip+0x17f12]        # 41a0e8 <_ZNSi10_M_extractIlEERSiRT_@GLIBCXX_3.4.9>
  4021d6:	68 1a 00 00 00                                  	push   0x1a
  4021db:	e9 40 fe ff ff                                  	jmp    402020 <.plt>

00000000004021e0 <_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base@plt>:
  4021e0:	ff 25 0a 7f 01 00                               	jmp    QWORD PTR [rip+0x17f0a]        # 41a0f0 <_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base@GLIBCXX_3.4>
  4021e6:	68 1b 00 00 00                                  	push   0x1b
  4021eb:	e9 30 fe ff ff                                  	jmp    402020 <.plt>

00000000004021f0 <_ZNSt11regex_errorD1Ev@plt>:
  4021f0:	ff 25 02 7f 01 00                               	jmp    QWORD PTR [rip+0x17f02]        # 41a0f8 <_ZNSt11regex_errorD1Ev@GLIBCXX_3.4.15>
  4021f6:	68 1c 00 00 00                                  	push   0x1c
  4021fb:	e9 20 fe ff ff                                  	jmp    402020 <.plt>

0000000000402200 <_ZNSt6localeaSERKS_@plt>:
  402200:	ff 25 fa 7e 01 00                               	jmp    QWORD PTR [rip+0x17efa]        # 41a100 <_ZNSt6localeaSERKS_@GLIBCXX_3.4>
  402206:	68 1d 00 00 00                                  	push   0x1d
  40220b:	e9 10 fe ff ff                                  	jmp    402020 <.plt>

0000000000402210 <__dynamic_cast@plt>:
  402210:	ff 25 f2 7e 01 00                               	jmp    QWORD PTR [rip+0x17ef2]        # 41a108 <__dynamic_cast@CXXABI_1.3>
  402216:	68 1e 00 00 00                                  	push   0x1e
  40221b:	e9 00 fe ff ff                                  	jmp    402020 <.plt>

0000000000402220 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>:
  402220:	ff 25 ea 7e 01 00                               	jmp    QWORD PTR [rip+0x17eea]        # 41a110 <_ZNKSt5ctypeIcE13_M_widen_initEv@GLIBCXX_3.4.11>
  402226:	68 1f 00 00 00                                  	push   0x1f
  40222b:	e9 f0 fd ff ff                                  	jmp    402020 <.plt>

0000000000402230 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc@plt>:
  402230:	ff 25 e2 7e 01 00                               	jmp    QWORD PTR [rip+0x17ee2]        # 41a118 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc@GLIBCXX_3.4.21>
  402236:	68 20 00 00 00                                  	push   0x20
  40223b:	e9 e0 fd ff ff                                  	jmp    402020 <.plt>

0000000000402240 <_ZSt16__throw_bad_castv@plt>:
  402240:	ff 25 da 7e 01 00                               	jmp    QWORD PTR [rip+0x17eda]        # 41a120 <_ZSt16__throw_bad_castv@GLIBCXX_3.4>
  402246:	68 21 00 00 00                                  	push   0x21
  40224b:	e9 d0 fd ff ff                                  	jmp    402020 <.plt>

0000000000402250 <_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE7_M_syncEPcmm@plt>:
  402250:	ff 25 d2 7e 01 00                               	jmp    QWORD PTR [rip+0x17ed2]        # 41a128 <_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE7_M_syncEPcmm@GLIBCXX_3.4.21>
  402256:	68 22 00 00 00                                  	push   0x22
  40225b:	e9 c0 fd ff ff                                  	jmp    402020 <.plt>

0000000000402260 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>:
  402260:	ff 25 ca 7e 01 00                               	jmp    QWORD PTR [rip+0x17eca]        # 41a130 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@GLIBCXX_3.4>
  402266:	68 23 00 00 00                                  	push   0x23
  40226b:	e9 b0 fd ff ff                                  	jmp    402020 <.plt>

0000000000402270 <strcmp@plt>:
  402270:	ff 25 c2 7e 01 00                               	jmp    QWORD PTR [rip+0x17ec2]        # 41a138 <strcmp@GLIBC_2.2.5>
  402276:	68 24 00 00 00                                  	push   0x24
  40227b:	e9 a0 fd ff ff                                  	jmp    402020 <.plt>

0000000000402280 <_ZNSt6localeD1Ev@plt>:
  402280:	ff 25 ba 7e 01 00                               	jmp    QWORD PTR [rip+0x17eba]        # 41a140 <_ZNSt6localeD1Ev@GLIBCXX_3.4>
  402286:	68 25 00 00 00                                  	push   0x25
  40228b:	e9 90 fd ff ff                                  	jmp    402020 <.plt>

0000000000402290 <__cxa_rethrow@plt>:
  402290:	ff 25 b2 7e 01 00                               	jmp    QWORD PTR [rip+0x17eb2]        # 41a148 <__cxa_rethrow@CXXABI_1.3>
  402296:	68 26 00 00 00                                  	push   0x26
  40229b:	e9 80 fd ff ff                                  	jmp    402020 <.plt>

00000000004022a0 <memmove@plt>:
  4022a0:	ff 25 aa 7e 01 00                               	jmp    QWORD PTR [rip+0x17eaa]        # 41a150 <memmove@GLIBC_2.2.5>
  4022a6:	68 27 00 00 00                                  	push   0x27
  4022ab:	e9 70 fd ff ff                                  	jmp    402020 <.plt>

00000000004022b0 <__cxa_end_catch@plt>:
  4022b0:	ff 25 a2 7e 01 00                               	jmp    QWORD PTR [rip+0x17ea2]        # 41a158 <__cxa_end_catch@CXXABI_1.3>
  4022b6:	68 28 00 00 00                                  	push   0x28
  4022bb:	e9 60 fd ff ff                                  	jmp    402020 <.plt>

00000000004022c0 <__gxx_personality_v0@plt>:
  4022c0:	ff 25 9a 7e 01 00                               	jmp    QWORD PTR [rip+0x17e9a]        # 41a160 <__gxx_personality_v0@CXXABI_1.3>
  4022c6:	68 29 00 00 00                                  	push   0x29
  4022cb:	e9 50 fd ff ff                                  	jmp    402020 <.plt>

00000000004022d0 <__cxa_throw@plt>:
  4022d0:	ff 25 92 7e 01 00                               	jmp    QWORD PTR [rip+0x17e92]        # 41a168 <__cxa_throw@CXXABI_1.3>
  4022d6:	68 2a 00 00 00                                  	push   0x2a
  4022db:	e9 40 fd ff ff                                  	jmp    402020 <.plt>

00000000004022e0 <_Unwind_Resume@plt>:
  4022e0:	ff 25 8a 7e 01 00                               	jmp    QWORD PTR [rip+0x17e8a]        # 41a170 <_Unwind_Resume@GCC_3.0>
  4022e6:	68 2b 00 00 00                                  	push   0x2b
  4022eb:	e9 30 fd ff ff                                  	jmp    402020 <.plt>

00000000004022f0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@plt>:
  4022f0:	ff 25 82 7e 01 00                               	jmp    QWORD PTR [rip+0x17e82]        # 41a178 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@GLIBCXX_3.4.21>
  4022f6:	68 2c 00 00 00                                  	push   0x2c
  4022fb:	e9 20 fd ff ff                                  	jmp    402020 <.plt>

0000000000402300 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm@plt>:
  402300:	ff 25 7a 7e 01 00                               	jmp    QWORD PTR [rip+0x17e7a]        # 41a180 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm@GLIBCXX_3.4.21>
  402306:	68 2d 00 00 00                                  	push   0x2d
  40230b:	e9 10 fd ff ff                                  	jmp    402020 <.plt>

0000000000402310 <__cxa_guard_acquire@plt>:
  402310:	ff 25 72 7e 01 00                               	jmp    QWORD PTR [rip+0x17e72]        # 41a188 <__cxa_guard_acquire@CXXABI_1.3>
  402316:	68 2e 00 00 00                                  	push   0x2e
  40231b:	e9 00 fd ff ff                                  	jmp    402020 <.plt>

0000000000402320 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc@plt>:
  402320:	ff 25 6a 7e 01 00                               	jmp    QWORD PTR [rip+0x17e6a]        # 41a190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc@GLIBCXX_3.4.21>
  402326:	68 2f 00 00 00                                  	push   0x2f
  40232b:	e9 f0 fc ff ff                                  	jmp    402020 <.plt>

0000000000402330 <_ZNSt13runtime_errorC2EPKc@plt>:
  402330:	ff 25 62 7e 01 00                               	jmp    QWORD PTR [rip+0x17e62]        # 41a198 <_ZNSt13runtime_errorC2EPKc@GLIBCXX_3.4.21>
  402336:	68 30 00 00 00                                  	push   0x30
  40233b:	e9 e0 fc ff ff                                  	jmp    402020 <.plt>

0000000000402340 <__cxa_bad_cast@plt>:
  402340:	ff 25 5a 7e 01 00                               	jmp    QWORD PTR [rip+0x17e5a]        # 41a1a0 <__cxa_bad_cast@CXXABI_1.3>
  402346:	68 31 00 00 00                                  	push   0x31
  40234b:	e9 d0 fc ff ff                                  	jmp    402020 <.plt>

0000000000402350 <_ZNSt6localeC1Ev@plt>:
  402350:	ff 25 52 7e 01 00                               	jmp    QWORD PTR [rip+0x17e52]        # 41a1a8 <_ZNSt6localeC1Ev@GLIBCXX_3.4>
  402356:	68 32 00 00 00                                  	push   0x32
  40235b:	e9 c0 fc ff ff                                  	jmp    402020 <.plt>

Disassembly of section .text:

0000000000402360 <_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc>:
_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_error.h:171
  402360:	41 55                                           	push   r13
  402362:	49 89 f5                                        	mov    r13,rsi
  402365:	41 54                                           	push   r12
  402367:	41 89 fc                                        	mov    r12d,edi
  40236a:	bf 18 00 00 00                                  	mov    edi,0x18
  40236f:	55                                              	push   rbp
  402370:	e8 6b fd ff ff                                  	call   4020e0 <__cxa_allocate_exception@plt>
_ZNSt11regex_errorC4ENSt15regex_constants10error_typeEPKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_error.h:158
  402375:	4c 89 ee                                        	mov    rsi,r13
  402378:	48 89 c7                                        	mov    rdi,rax
_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_error.h:171
  40237b:	48 89 c5                                        	mov    rbp,rax
_ZNSt11regex_errorC4ENSt15regex_constants10error_typeEPKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_error.h:158
  40237e:	e8 ad ff ff ff                                  	call   402330 <_ZNSt13runtime_errorC2EPKc@plt>
_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_error.h:171
  402383:	ba f0 21 40 00                                  	mov    edx,0x4021f0
  402388:	be 20 9d 41 00                                  	mov    esi,0x419d20
  40238d:	48 89 ef                                        	mov    rdi,rbp
_ZNSt11regex_errorC4ENSt15regex_constants10error_typeEPKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_error.h:158
  402390:	48 c7 45 00 f0 9a 41 00                         	mov    QWORD PTR [rbp+0x0],0x419af0
  402398:	44 89 65 10                                     	mov    DWORD PTR [rbp+0x10],r12d
_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_error.h:171
  40239c:	e8 2f ff ff ff                                  	call   4022d0 <__cxa_throw@plt>
  4023a1:	49 89 c4                                        	mov    r12,rax
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_error.h:171 (discriminator 1)
  4023a4:	48 89 ef                                        	mov    rdi,rbp
  4023a7:	e8 c4 fd ff ff                                  	call   402170 <__cxa_free_exception@plt>
  4023ac:	4c 89 e7                                        	mov    rdi,r12
  4023af:	e8 2c ff ff ff                                  	call   4022e0 <_Unwind_Resume@plt>

00000000004023b4 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0.cold>:
_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/sstream:65
  4023b4:	48 c7 44 24 40 48 9d 41 00                      	mov    QWORD PTR [rsp+0x40],0x419d48
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  4023bd:	48 8b bc 24 88 00 00 00                         	mov    rdi,QWORD PTR [rsp+0x88]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:231
  4023c5:	4c 39 ef                                        	cmp    rdi,r13
  4023c8:	74 11                                           	je     4023db <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0.cold+0x27>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:237
  4023ca:	48 8b 84 24 98 00 00 00                         	mov    rax,QWORD PTR [rsp+0x98]
  4023d2:	48 8d 70 01                                     	lea    rsi,[rax+0x1]
_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  4023d6:	e8 e5 fd ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt15basic_streambufIcSt11char_traitsIcEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/streambuf:205
  4023db:	48 c7 44 24 40 88 9b 41 00                      	mov    QWORD PTR [rsp+0x40],0x419b88
  4023e4:	48 8d 7c 24 78                                  	lea    rdi,[rsp+0x78]
  4023e9:	e8 92 fe ff ff                                  	call   402280 <_ZNSt6localeD1Ev@plt>
_ZNSiD4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/istream:104
  4023ee:	48 8b 43 e8                                     	mov    rax,QWORD PTR [rbx-0x18]
  4023f2:	48 89 5c 24 30                                  	mov    QWORD PTR [rsp+0x30],rbx
  4023f7:	48 8b 0d 7a 78 01 00                            	mov    rcx,QWORD PTR [rip+0x1787a]        # 419c78 <_ZTTNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEE@@GLIBCXX_3.4.21+0x10>
  4023fe:	48 89 4c 04 30                                  	mov    QWORD PTR [rsp+rax*1+0x30],rcx
  402403:	48 c7 44 24 38 00 00 00 00                      	mov    QWORD PTR [rsp+0x38],0x0
_ZNSt9basic_iosIcSt11char_traitsIcEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_ios.h:282
  40240c:	48 8d bc 24 a8 00 00 00                         	lea    rdi,[rsp+0xa8]
  402414:	48 c7 84 24 a8 00 00 00 18 9b 41 00             	mov    QWORD PTR [rsp+0xa8],0x419b18
  402420:	e8 4b fc ff ff                                  	call   402070 <_ZNSt8ios_baseD2Ev@plt>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402425:	48 8b 7c 24 10                                  	mov    rdi,QWORD PTR [rsp+0x10]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:231
  40242a:	4c 39 ff                                        	cmp    rdi,r15
  40242d:	74 0e                                           	je     40243d <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0.cold+0x89>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:237
  40242f:	48 8b 44 24 20                                  	mov    rax,QWORD PTR [rsp+0x20]
  402434:	48 8d 70 01                                     	lea    rsi,[rax+0x1]
_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  402438:	e8 83 fd ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZN9__gnu_cxx13new_allocatorIcED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:89
  40243d:	48 89 ef                                        	mov    rdi,rbp
  402440:	e8 9b fe ff ff                                  	call   4022e0 <_Unwind_Resume@plt>
_ZNKSt7__cxx1112regex_traitsIcE5valueEci():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.tcc:341
  402445:	48 8d 7c 24 30                                  	lea    rdi,[rsp+0x30]
  40244a:	e8 81 fc ff ff                                  	call   4020d0 <_ZNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEED1Ev@plt>
  40244f:	48 89 ef                                        	mov    rdi,rbp
  402452:	e8 89 fe ff ff                                  	call   4022e0 <_Unwind_Resume@plt>

0000000000402457 <_Z9regexTestv.cold>:
_ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEEC4IPKcEET_S7_St6localeNSt15regex_constants18syntax_option_typeE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.tcc:341
  402457:	48 8d 7c 24 38                                  	lea    rdi,[rsp+0x38]
  40245c:	e8 1f fe ff ff                                  	call   402280 <_ZNSt6localeD1Ev@plt>
_ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEEC4IPKcEET_S7_NSt15regex_constants18syntax_option_typeE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:507
  402461:	48 8d 7c 24 08                                  	lea    rdi,[rsp+0x8]
  402466:	e8 15 fe ff ff                                  	call   402280 <_ZNSt6localeD1Ev@plt>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  40246b:	48 8b 7c 24 10                                  	mov    rdi,QWORD PTR [rsp+0x10]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:231
  402470:	4c 39 e7                                        	cmp    rdi,r12
  402473:	74 0e                                           	je     402483 <_Z9regexTestv.cold+0x2c>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:237
  402475:	48 8b 44 24 20                                  	mov    rax,QWORD PTR [rsp+0x20]
  40247a:	48 8d 70 01                                     	lea    rsi,[rax+0x1]
_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  40247e:	e8 3d fd ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZN9__gnu_cxx13new_allocatorIcED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:89
  402483:	48 89 ef                                        	mov    rdi,rbp
  402486:	e8 55 fe ff ff                                  	call   4022e0 <_Unwind_Resume@plt>
_Z9regexTestv.cold():
  40248b:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]

0000000000402490 <main>:
main():
/tmp/compiler-explorer-compiler202095-9137-1kkxb4c.yd7w/example.cpp:12
  402490:	48 83 ec 08                                     	sub    rsp,0x8
/tmp/compiler-explorer-compiler202095-9137-1kkxb4c.yd7w/example.cpp:13
  402494:	e8 e7 08 00 00                                  	call   402d80 <_Z9regexTestv>
/tmp/compiler-explorer-compiler202095-9137-1kkxb4c.yd7w/example.cpp:15
  402499:	b8 2a 00 00 00                                  	mov    eax,0x2a
  40249e:	48 83 c4 08                                     	add    rsp,0x8
  4024a2:	c3                                              	ret
  4024a3:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  4024ad:	0f 1f 00                                        	nop    DWORD PTR [rax]

00000000004024b0 <_start>:
_start():
  4024b0:	f3 0f 1e fa                                     	endbr64
  4024b4:	31 ed                                           	xor    ebp,ebp
  4024b6:	49 89 d1                                        	mov    r9,rdx
  4024b9:	5e                                              	pop    rsi
  4024ba:	48 89 e2                                        	mov    rdx,rsp
  4024bd:	48 83 e4 f0                                     	and    rsp,0xfffffffffffffff0
  4024c1:	50                                              	push   rax
  4024c2:	54                                              	push   rsp
  4024c3:	49 c7 c0 30 36 41 00                            	mov    r8,0x413630
  4024ca:	48 c7 c1 c0 35 41 00                            	mov    rcx,0x4135c0
  4024d1:	48 c7 c7 90 24 40 00                            	mov    rdi,0x402490
  4024d8:	ff 15 fa 7a 01 00                               	call   QWORD PTR [rip+0x17afa]        # 419fd8 <__libc_start_main@GLIBC_2.2.5>
  4024de:	f4                                              	hlt
  4024df:	90                                              	nop

00000000004024e0 <_dl_relocate_static_pie>:
_dl_relocate_static_pie():
  4024e0:	f3 0f 1e fa                                     	endbr64
  4024e4:	c3                                              	ret
  4024e5:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  4024ef:	90                                              	nop

00000000004024f0 <deregister_tm_clones>:
deregister_tm_clones():
  4024f0:	b8 e0 a5 41 00                                  	mov    eax,0x41a5e0
  4024f5:	48 3d e0 a5 41 00                               	cmp    rax,0x41a5e0
  4024fb:	74 13                                           	je     402510 <deregister_tm_clones+0x20>
  4024fd:	b8 00 00 00 00                                  	mov    eax,0x0
  402502:	48 85 c0                                        	test   rax,rax
  402505:	74 09                                           	je     402510 <deregister_tm_clones+0x20>
  402507:	bf e0 a5 41 00                                  	mov    edi,0x41a5e0
  40250c:	ff e0                                           	jmp    rax
  40250e:	66 90                                           	xchg   ax,ax
  402510:	c3                                              	ret
  402511:	66 66 2e 0f 1f 84 00 00 00 00 00                	data16 nop WORD PTR cs:[rax+rax*1+0x0]
  40251c:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

0000000000402520 <register_tm_clones>:
register_tm_clones():
  402520:	be e0 a5 41 00                                  	mov    esi,0x41a5e0
  402525:	48 81 ee e0 a5 41 00                            	sub    rsi,0x41a5e0
  40252c:	48 89 f0                                        	mov    rax,rsi
  40252f:	48 c1 ee 3f                                     	shr    rsi,0x3f
  402533:	48 c1 f8 03                                     	sar    rax,0x3
  402537:	48 01 c6                                        	add    rsi,rax
  40253a:	48 d1 fe                                        	sar    rsi,1
  40253d:	74 11                                           	je     402550 <register_tm_clones+0x30>
  40253f:	b8 00 00 00 00                                  	mov    eax,0x0
  402544:	48 85 c0                                        	test   rax,rax
  402547:	74 07                                           	je     402550 <register_tm_clones+0x30>
  402549:	bf e0 a5 41 00                                  	mov    edi,0x41a5e0
  40254e:	ff e0                                           	jmp    rax
  402550:	c3                                              	ret
  402551:	66 66 2e 0f 1f 84 00 00 00 00 00                	data16 nop WORD PTR cs:[rax+rax*1+0x0]
  40255c:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

0000000000402560 <__do_global_dtors_aux>:
__do_global_dtors_aux():
  402560:	80 3d 81 80 01 00 00                            	cmp    BYTE PTR [rip+0x18081],0x0        # 41a5e8 <completed.0>
  402567:	75 17                                           	jne    402580 <__do_global_dtors_aux+0x20>
  402569:	55                                              	push   rbp
  40256a:	48 89 e5                                        	mov    rbp,rsp
  40256d:	e8 7e ff ff ff                                  	call   4024f0 <deregister_tm_clones>
  402572:	c6 05 6f 80 01 00 01                            	mov    BYTE PTR [rip+0x1806f],0x1        # 41a5e8 <completed.0>
  402579:	5d                                              	pop    rbp
  40257a:	c3                                              	ret
  40257b:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]
  402580:	c3                                              	ret
  402581:	66 66 2e 0f 1f 84 00 00 00 00 00                	data16 nop WORD PTR cs:[rax+rax*1+0x0]
  40258c:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

0000000000402590 <frame_dummy>:
frame_dummy():
  402590:	eb 8e                                           	jmp    402520 <register_tm_clones>
  402592:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  40259c:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

00000000004025a0 <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0>:
_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:228
  4025a0:	48 8d 42 ff                                     	lea    rax,[rdx-0x1]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:223
  4025a4:	41 56                                           	push   r14
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:237
  4025a6:	49 89 d6                                        	mov    r14,rdx
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:223
  4025a9:	49 89 f0                                        	mov    r8,rsi
  4025ac:	41 55                                           	push   r13
  4025ae:	41 89 ca                                        	mov    r10d,ecx
  4025b1:	49 89 d5                                        	mov    r13,rdx
  4025b4:	41 89 c9                                        	mov    r9d,ecx
  4025b7:	41 54                                           	push   r12
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:228
  4025b9:	49 89 c4                                        	mov    r12,rax
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:237
  4025bc:	41 83 e6 01                                     	and    r14d,0x1
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:228
  4025c0:	49 c1 ec 3f                                     	shr    r12,0x3f
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:223
  4025c4:	55                                              	push   rbp
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:228
  4025c5:	49 01 c4                                        	add    r12,rax
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:223
  4025c8:	53                                              	push   rbx
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:228
  4025c9:	49 d1 fc                                        	sar    r12,1
  4025cc:	4c 39 e6                                        	cmp    rsi,r12
  4025cf:	7c 1a                                           	jl     4025eb <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0x4b>
  4025d1:	e9 ba 00 00 00                                  	jmp    402690 <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0xf0>
  4025d6:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:234
  4025e0:	88 1c 37                                        	mov    BYTE PTR [rdi+rsi*1],bl
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:228
  4025e3:	49 39 c4                                        	cmp    r12,rax
  4025e6:	7e 3e                                           	jle    402626 <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0x86>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:223
  4025e8:	48 89 c6                                        	mov    rsi,rax
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:230
  4025eb:	48 8d 44 36 02                                  	lea    rax,[rsi+rsi*1+0x2]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:231
  4025f0:	48 8d 48 ff                                     	lea    rcx,[rax-0x1]
_ZNK9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEplEl():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1017
  4025f4:	48 8d 14 07                                     	lea    rdx,[rdi+rax*1]
  4025f8:	48 8d 2c 0f                                     	lea    rbp,[rdi+rcx*1]
_ZNK9__gnu_cxx5__ops15_Iter_less_iterclINS_17__normal_iteratorIPcSt6vectorIcSaIcEEEES8_EEbT_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/predefined_ops.h:43
  4025fc:	0f b6 1a                                        	movzx  ebx,BYTE PTR [rdx]
  4025ff:	44 0f b6 5d 00                                  	movzx  r11d,BYTE PTR [rbp+0x0]
_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:231
  402604:	41 38 db                                        	cmp    r11b,bl
  402607:	7e d7                                           	jle    4025e0 <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0x40>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:234
  402609:	44 88 1c 37                                     	mov    BYTE PTR [rdi+rsi*1],r11b
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:228
  40260d:	49 39 cc                                        	cmp    r12,rcx
  402610:	7e 0e                                           	jle    402620 <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0x80>
  402612:	48 89 c8                                        	mov    rax,rcx
  402615:	eb d1                                           	jmp    4025e8 <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0x48>
  402617:	66 0f 1f 84 00 00 00 00 00                      	nop    WORD PTR [rax+rax*1+0x0]
  402620:	48 89 ea                                        	mov    rdx,rbp
  402623:	48 89 c8                                        	mov    rax,rcx
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:237
  402626:	4d 85 f6                                        	test   r14,r14
  402629:	74 75                                           	je     4026a0 <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0x100>
_ZSt11__push_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops14_Iter_less_valEEvT_T0_SA_T1_RT2_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:138
  40262b:	48 8d 70 ff                                     	lea    rsi,[rax-0x1]
  40262f:	48 89 f1                                        	mov    rcx,rsi
  402632:	48 c1 e9 3f                                     	shr    rcx,0x3f
  402636:	48 01 f1                                        	add    rcx,rsi
  402639:	48 d1 f9                                        	sar    rcx,1
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:139
  40263c:	4c 39 c0                                        	cmp    rax,r8
  40263f:	7f 29                                           	jg     40266a <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0xca>
  402641:	eb 38                                           	jmp    40267b <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0xdb>
  402643:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:141
  402648:	40 88 32                                        	mov    BYTE PTR [rdx],sil
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:143
  40264b:	48 8d 51 ff                                     	lea    rdx,[rcx-0x1]
  40264f:	48 89 d0                                        	mov    rax,rdx
  402652:	48 c1 e8 3f                                     	shr    rax,0x3f
  402656:	48 01 d0                                        	add    rax,rdx
  402659:	48 d1 f8                                        	sar    rax,1
  40265c:	48 89 c2                                        	mov    rdx,rax
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:139
  40265f:	48 89 c8                                        	mov    rax,rcx
  402662:	49 39 c8                                        	cmp    r8,rcx
  402665:	7d 71                                           	jge    4026d8 <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0x138>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:143
  402667:	48 89 d1                                        	mov    rcx,rdx
_ZNK9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEplEl():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1017
  40266a:	4c 8d 1c 0f                                     	lea    r11,[rdi+rcx*1]
  40266e:	48 8d 14 07                                     	lea    rdx,[rdi+rax*1]
_ZNK9__gnu_cxx5__ops14_Iter_less_valclINS_17__normal_iteratorIPcSt6vectorIcSaIcEEEEcEEbT_RT0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/predefined_ops.h:67
  402672:	41 0f b6 33                                     	movzx  esi,BYTE PTR [r11]
_ZSt11__push_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops14_Iter_less_valEEvT_T0_SA_T1_RT2_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:139
  402676:	44 38 ce                                        	cmp    sil,r9b
  402679:	7c cd                                           	jl     402648 <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0xa8>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:145
  40267b:	44 88 12                                        	mov    BYTE PTR [rdx],r10b
_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:248
  40267e:	5b                                              	pop    rbx
  40267f:	5d                                              	pop    rbp
  402680:	41 5c                                           	pop    r12
  402682:	41 5d                                           	pop    r13
  402684:	41 5e                                           	pop    r14
  402686:	c3                                              	ret
  402687:	66 0f 1f 84 00 00 00 00 00                      	nop    WORD PTR [rax+rax*1+0x0]
_ZNK9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEplEl():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1017
  402690:	48 8d 14 37                                     	lea    rdx,[rdi+rsi*1]
_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:237
  402694:	4d 85 f6                                        	test   r14,r14
  402697:	75 e2                                           	jne    40267b <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0xdb>
  402699:	4c 89 c0                                        	mov    rax,r8
  40269c:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]
  4026a0:	49 8d 4d fe                                     	lea    rcx,[r13-0x2]
  4026a4:	49 89 cd                                        	mov    r13,rcx
  4026a7:	49 c1 ed 3f                                     	shr    r13,0x3f
  4026ab:	49 01 cd                                        	add    r13,rcx
  4026ae:	49 d1 fd                                        	sar    r13,1
  4026b1:	4c 39 e8                                        	cmp    rax,r13
  4026b4:	0f 85 71 ff ff ff                               	jne    40262b <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0x8b>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:240
  4026ba:	48 8d 44 00 01                                  	lea    rax,[rax+rax*1+0x1]
_ZNK9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEplEl():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1017
  4026bf:	48 8d 0c 07                                     	lea    rcx,[rdi+rax*1]
_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:240
  4026c3:	0f b6 31                                        	movzx  esi,BYTE PTR [rcx]
  4026c6:	40 88 32                                        	mov    BYTE PTR [rdx],sil
  4026c9:	48 89 ca                                        	mov    rdx,rcx
  4026cc:	e9 5a ff ff ff                                  	jmp    40262b <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0+0x8b>
  4026d1:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
_ZSt11__push_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops14_Iter_less_valEEvT_T0_SA_T1_RT2_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:240
  4026d8:	4c 89 da                                        	mov    rdx,r11
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:145
  4026db:	44 88 12                                        	mov    BYTE PTR [rdx],r10b
_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:248
  4026de:	5b                                              	pop    rbx
  4026df:	5d                                              	pop    rbp
  4026e0:	41 5c                                           	pop    r12
  4026e2:	41 5d                                           	pop    r13
  4026e4:	41 5e                                           	pop    r14
  4026e6:	c3                                              	ret
_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:248
  4026e7:	90                                              	nop
  4026e8:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]

00000000004026f0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0>:
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:703
  4026f0:	55                                              	push   rbp
  4026f1:	48 89 fd                                        	mov    rbp,rdi
  4026f4:	53                                              	push   rbx
  4026f5:	48 89 f3                                        	mov    rbx,rsi
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:203
  4026f8:	48 8d 43 10                                     	lea    rax,[rbx+0x10]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:703
  4026fc:	48 83 ec 08                                     	sub    rsp,0x8
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402700:	48 8b 36                                        	mov    rsi,QWORD PTR [rsi]
  402703:	48 8b 3f                                        	mov    rdi,QWORD PTR [rdi]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:902
  402706:	48 8b 53 08                                     	mov    rdx,QWORD PTR [rbx+0x8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:718
  40270a:	48 39 c6                                        	cmp    rsi,rax
  40270d:	74 61                                           	je     402770 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0+0x80>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:203
  40270f:	48 8d 4d 10                                     	lea    rcx,[rbp+0x10]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:732
  402713:	48 39 cf                                        	cmp    rdi,rcx
  402716:	74 38                                           	je     402750 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0+0x60>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:179
  402718:	48 89 75 00                                     	mov    QWORD PTR [rbp+0x0],rsi
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:738
  40271c:	48 8b 4d 10                                     	mov    rcx,QWORD PTR [rbp+0x10]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:183
  402720:	48 89 55 08                                     	mov    QWORD PTR [rbp+0x8],rdx
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:746
  402724:	48 8b 53 10                                     	mov    rdx,QWORD PTR [rbx+0x10]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:211
  402728:	48 89 55 10                                     	mov    QWORD PTR [rbp+0x10],rdx
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:747
  40272c:	48 85 ff                                        	test   rdi,rdi
  40272f:	74 2f                                           	je     402760 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0+0x70>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:179
  402731:	48 89 3b                                        	mov    QWORD PTR [rbx],rdi
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:211
  402734:	48 89 4b 10                                     	mov    QWORD PTR [rbx+0x10],rcx
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:183
  402738:	48 c7 43 08 00 00 00 00                         	mov    QWORD PTR [rbx+0x8],0x0
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  402740:	c6 07 00                                        	mov    BYTE PTR [rdi],0x0
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:759
  402743:	48 83 c4 08                                     	add    rsp,0x8
  402747:	5b                                              	pop    rbx
  402748:	5d                                              	pop    rbp
  402749:	c3                                              	ret
  40274a:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:179
  402750:	48 89 75 00                                     	mov    QWORD PTR [rbp+0x0],rsi
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:183
  402754:	48 89 55 08                                     	mov    QWORD PTR [rbp+0x8],rdx
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:746
  402758:	48 8b 53 10                                     	mov    rdx,QWORD PTR [rbx+0x10]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:211
  40275c:	48 89 55 10                                     	mov    QWORD PTR [rbp+0x10],rdx
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:179
  402760:	48 89 03                                        	mov    QWORD PTR [rbx],rax
  402763:	48 89 c7                                        	mov    rdi,rax
  402766:	eb d0                                           	jmp    402738 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0+0x48>
  402768:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:721
  402770:	48 85 d2                                        	test   rdx,rdx
  402773:	74 13                                           	je     402788 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0+0x98>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:348
  402775:	48 83 fa 01                                     	cmp    rdx,0x1
  402779:	74 1d                                           	je     402798 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0+0xa8>
_ZNSt11char_traitsIcE4copyEPcPKcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:395
  40277b:	e8 00 fa ff ff                                  	call   402180 <memcpy@plt>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:902
  402780:	48 8b 53 08                                     	mov    rdx,QWORD PTR [rbx+0x8]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402784:	48 8b 7d 00                                     	mov    rdi,QWORD PTR [rbp+0x0]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:183
  402788:	48 89 55 08                                     	mov    QWORD PTR [rbp+0x8],rdx
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  40278c:	c6 04 17 00                                     	mov    BYTE PTR [rdi+rdx*1],0x0
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402790:	48 8b 3b                                        	mov    rdi,QWORD PTR [rbx]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:218
  402793:	eb a3                                           	jmp    402738 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0+0x48>
  402795:	0f 1f 00                                        	nop    DWORD PTR [rax]
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  402798:	0f b6 43 10                                     	movzx  eax,BYTE PTR [rbx+0x10]
  40279c:	88 07                                           	mov    BYTE PTR [rdi],al
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:902
  40279e:	48 8b 53 08                                     	mov    rdx,QWORD PTR [rbx+0x8]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  4027a2:	48 8b 7d 00                                     	mov    rdi,QWORD PTR [rbp+0x0]
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  4027a6:	eb e0                                           	jmp    402788 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0+0x98>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_.isra.0():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  4027a8:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]

00000000004027b0 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0>:
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1945
  4027b0:	41 57                                           	push   r15
_ZN9__gnu_cxxmiIPcSt6vectorIcSaIcEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1180
  4027b2:	49 89 f7                                        	mov    r15,rsi
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1945
  4027b5:	41 56                                           	push   r14
_ZN9__gnu_cxxmiIPcSt6vectorIcSaIcEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1180
  4027b7:	49 29 ff                                        	sub    r15,rdi
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1945
  4027ba:	41 55                                           	push   r13
  4027bc:	41 54                                           	push   r12
  4027be:	55                                              	push   rbp
  4027bf:	53                                              	push   rbx
  4027c0:	48 83 ec 08                                     	sub    rsp,0x8
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1949
  4027c4:	49 83 ff 10                                     	cmp    r15,0x10
  4027c8:	0f 8e 6e 01 00 00                               	jle    40293c <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x18c>
  4027ce:	49 89 fd                                        	mov    r13,rdi
  4027d1:	49 89 d6                                        	mov    r14,rdx
  4027d4:	49 89 f4                                        	mov    r12,rsi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1951
  4027d7:	48 85 d2                                        	test   rdx,rdx
  4027da:	0f 84 00 01 00 00                               	je     4028e0 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x130>
  4027e0:	49 89 f0                                        	mov    r8,rsi
_ZNK9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEplEl():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1017
  4027e3:	48 8d 6f 01                                     	lea    rbp,[rdi+0x1]
_ZN9__gnu_cxxmiIPcSt6vectorIcSaIcEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1180
  4027e7:	4c 89 c2                                        	mov    rdx,r8
  4027ea:	41 0f b7 75 00                                  	movzx  esi,WORD PTR [r13+0x0]
_ZNK9__gnu_cxx5__ops15_Iter_less_iterclINS_17__normal_iteratorIPcSt6vectorIcSaIcEEEES8_EEbT_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/predefined_ops.h:43
  4027ef:	41 0f b6 4d 01                                  	movzx  ecx,BYTE PTR [r13+0x1]
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1956
  4027f4:	49 83 ee 01                                     	sub    r14,0x1
_ZN9__gnu_cxxmiIPcSt6vectorIcSaIcEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1180
  4027f8:	4c 29 ea                                        	sub    rdx,r13
_ZNK9__gnu_cxx5__ops15_Iter_less_iterclINS_17__normal_iteratorIPcSt6vectorIcSaIcEEEES8_EEbT_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/predefined_ops.h:43
  4027fb:	45 0f b6 48 ff                                  	movzx  r9d,BYTE PTR [r8-0x1]
_ZSt27__unguarded_partition_pivotIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEET_S9_S9_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1923
  402800:	48 89 d0                                        	mov    rax,rdx
  402803:	66 c1 c6 08                                     	rol    si,0x8
  402807:	48 c1 e8 3f                                     	shr    rax,0x3f
  40280b:	48 01 d0                                        	add    rax,rdx
_ZSt4swapIcENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS4_ESt18is_move_assignableIS4_EEE5valueEvE4typeERS4_SE_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:197
  40280e:	41 0f b6 55 00                                  	movzx  edx,BYTE PTR [r13+0x0]
_ZSt27__unguarded_partition_pivotIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEET_S9_S9_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1923
  402813:	48 d1 f8                                        	sar    rax,1
_ZNK9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEplEl():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1017
  402816:	4c 01 e8                                        	add    rax,r13
_ZNK9__gnu_cxx5__ops15_Iter_less_iterclINS_17__normal_iteratorIPcSt6vectorIcSaIcEEEES8_EEbT_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/predefined_ops.h:43
  402819:	0f b6 38                                        	movzx  edi,BYTE PTR [rax]
_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_S9_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:82
  40281c:	40 38 f9                                        	cmp    cl,dil
  40281f:	7d 5c                                           	jge    40287d <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xcd>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:84
  402821:	44 38 cf                                        	cmp    dil,r9b
  402824:	0f 8c a1 00 00 00                               	jl     4028cb <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x11b>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:86
  40282a:	44 38 c9                                        	cmp    cl,r9b
  40282d:	7c 58                                           	jl     402887 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xd7>
_ZSt4swapIcENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS4_ESt18is_move_assignableIS4_EEE5valueEvE4typeERS4_SE_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:198
  40282f:	66 41 89 75 00                                  	mov    WORD PTR [r13+0x0],si
_ZNK9__gnu_cxx5__ops15_Iter_less_iterclINS_17__normal_iteratorIPcSt6vectorIcSaIcEEEES8_EEbT_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/predefined_ops.h:43
  402834:	41 0f b6 50 ff                                  	movzx  edx,BYTE PTR [r8-0x1]
_ZSt21__unguarded_partitionIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEET_S9_S9_S9_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1898
  402839:	48 89 eb                                        	mov    rbx,rbp
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1898
  40283c:	4c 89 c0                                        	mov    rax,r8
  40283f:	90                                              	nop
_ZNK9__gnu_cxx5__ops15_Iter_less_iterclINS_17__normal_iteratorIPcSt6vectorIcSaIcEEEES8_EEbT_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/predefined_ops.h:43
  402840:	0f b6 33                                        	movzx  esi,BYTE PTR [rbx]
  402843:	49 89 dc                                        	mov    r12,rbx
_ZSt21__unguarded_partitionIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEET_S9_S9_S9_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1904
  402846:	40 38 f1                                        	cmp    cl,sil
  402849:	7f 2c                                           	jg     402877 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xc7>
_ZN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEmmEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:994
  40284b:	48 83 e8 01                                     	sub    rax,0x1
_ZSt21__unguarded_partitionIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEET_S9_S9_S9_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1907
  40284f:	38 ca                                           	cmp    dl,cl
  402851:	7e 11                                           	jle    402864 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xb4>
  402853:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]
_ZNK9__gnu_cxx5__ops15_Iter_less_iterclINS_17__normal_iteratorIPcSt6vectorIcSaIcEEEES8_EEbT_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/predefined_ops.h:43
  402858:	0f b6 50 ff                                     	movzx  edx,BYTE PTR [rax-0x1]
_ZN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEmmEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:994
  40285c:	48 83 e8 01                                     	sub    rax,0x1
_ZSt21__unguarded_partitionIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEET_S9_S9_S9_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1907
  402860:	38 d1                                           	cmp    cl,dl
  402862:	7c f4                                           	jl     402858 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xa8>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1909
  402864:	48 39 d8                                        	cmp    rax,rbx
  402867:	76 37                                           	jbe    4028a0 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xf0>
_ZSt4swapIcENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS4_ESt18is_move_assignableIS4_EEE5valueEvE4typeERS4_SE_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:198
  402869:	88 13                                           	mov    BYTE PTR [rbx],dl
_ZNK9__gnu_cxx5__ops15_Iter_less_iterclINS_17__normal_iteratorIPcSt6vectorIcSaIcEEEES8_EEbT_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/predefined_ops.h:43
  40286b:	0f b6 50 ff                                     	movzx  edx,BYTE PTR [rax-0x1]
_ZSt4swapIcENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS4_ESt18is_move_assignableIS4_EEE5valueEvE4typeERS4_SE_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:199
  40286f:	40 88 30                                        	mov    BYTE PTR [rax],sil
_ZNK9__gnu_cxx5__ops15_Iter_less_iterclINS_17__normal_iteratorIPcSt6vectorIcSaIcEEEES8_EEbT_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/predefined_ops.h:43
  402872:	41 0f b6 4d 00                                  	movzx  ecx,BYTE PTR [r13+0x0]
  402877:	48 83 c3 01                                     	add    rbx,0x1
  40287b:	eb c3                                           	jmp    402840 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x90>
_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_S9_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:91
  40287d:	44 38 c9                                        	cmp    cl,r9b
  402880:	7c ad                                           	jl     40282f <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x7f>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:93
  402882:	44 38 cf                                        	cmp    dil,r9b
  402885:	7d 44                                           	jge    4028cb <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x11b>
_ZSt4swapIcENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS4_ESt18is_move_assignableIS4_EEE5valueEvE4typeERS4_SE_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:198
  402887:	45 88 4d 00                                     	mov    BYTE PTR [r13+0x0],r9b
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:199
  40288b:	41 88 50 ff                                     	mov    BYTE PTR [r8-0x1],dl
_ZNK9__gnu_cxx5__ops15_Iter_less_iterclINS_17__normal_iteratorIPcSt6vectorIcSaIcEEEES8_EEbT_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/predefined_ops.h:43
  40288f:	41 0f b6 4d 00                                  	movzx  ecx,BYTE PTR [r13+0x0]
_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_S9_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/predefined_ops.h:43
  402894:	eb a3                                           	jmp    402839 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x89>
  402896:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
_ZN9__gnu_cxxmiIPcSt6vectorIcSaIcEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1180
  4028a0:	49 89 df                                        	mov    r15,rbx
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1959
  4028a3:	4c 89 f2                                        	mov    rdx,r14
  4028a6:	4c 89 c6                                        	mov    rsi,r8
  4028a9:	48 89 df                                        	mov    rdi,rbx
  4028ac:	e8 ff fe ff ff                                  	call   4027b0 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0>
_ZN9__gnu_cxxmiIPcSt6vectorIcSaIcEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1180
  4028b1:	4d 29 ef                                        	sub    r15,r13
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1949
  4028b4:	49 83 ff 10                                     	cmp    r15,0x10
  4028b8:	0f 8e 7e 00 00 00                               	jle    40293c <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x18c>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1951
  4028be:	4d 85 f6                                        	test   r14,r14
  4028c1:	74 1d                                           	je     4028e0 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x130>
  4028c3:	49 89 d8                                        	mov    r8,rbx
  4028c6:	e9 1c ff ff ff                                  	jmp    4027e7 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x37>
_ZSt4swapIcENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS4_ESt18is_move_assignableIS4_EEE5valueEvE4typeERS4_SE_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:198
  4028cb:	41 88 7d 00                                     	mov    BYTE PTR [r13+0x0],dil
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:199
  4028cf:	88 10                                           	mov    BYTE PTR [rax],dl
_ZNK9__gnu_cxx5__ops15_Iter_less_iterclINS_17__normal_iteratorIPcSt6vectorIcSaIcEEEES8_EEbT_T0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/predefined_ops.h:43
  4028d1:	41 0f b6 4d 00                                  	movzx  ecx,BYTE PTR [r13+0x0]
  4028d6:	41 0f b6 50 ff                                  	movzx  edx,BYTE PTR [r8-0x1]
  4028db:	e9 59 ff ff ff                                  	jmp    402839 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x89>
_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:351
  4028e0:	49 8d 5f fe                                     	lea    rbx,[r15-0x2]
  4028e4:	48 d1 fb                                        	sar    rbx,1
  4028e7:	eb 04                                           	jmp    4028ed <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x13d>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:359
  4028e9:	48 83 eb 01                                     	sub    rbx,0x1
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:355
  4028ed:	41 0f be 4c 1d 00                               	movsx  ecx,BYTE PTR [r13+rbx*1+0x0]
  4028f3:	4c 89 fa                                        	mov    rdx,r15
  4028f6:	48 89 de                                        	mov    rsi,rbx
  4028f9:	4c 89 ef                                        	mov    rdi,r13
  4028fc:	e8 9f fc ff ff                                  	call   4025a0 <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:357
  402901:	48 85 db                                        	test   rbx,rbx
  402904:	75 e3                                           	jne    4028e9 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x139>
_ZSt11__sort_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:421
  402906:	49 83 ec 01                                     	sub    r12,0x1
  40290a:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_RT0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:262
  402910:	41 0f b6 45 00                                  	movzx  eax,BYTE PTR [r13+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:261
  402915:	41 0f be 0c 24                                  	movsx  ecx,BYTE PTR [r12]
_ZN9__gnu_cxxmiIPcSt6vectorIcSaIcEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1180
  40291a:	4c 89 e3                                        	mov    rbx,r12
_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_RT0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:263
  40291d:	31 f6                                           	xor    esi,esi
_ZN9__gnu_cxxmiIPcSt6vectorIcSaIcEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1180
  40291f:	4c 29 eb                                        	sub    rbx,r13
_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_RT0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:263
  402922:	4c 89 ef                                        	mov    rdi,r13
_ZSt11__sort_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:421
  402925:	49 83 ec 01                                     	sub    r12,0x1
_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_RT0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:262
  402929:	41 88 44 24 01                                  	mov    BYTE PTR [r12+0x1],al
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:263
  40292e:	48 89 da                                        	mov    rdx,rbx
  402931:	e8 6a fc ff ff                                  	call   4025a0 <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElcNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0>
_ZSt11__sort_heapIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_heap.h:421
  402936:	48 83 fb 01                                     	cmp    rbx,0x1
  40293a:	7f d4                                           	jg     402910 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x160>
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1962
  40293c:	48 83 c4 08                                     	add    rsp,0x8
  402940:	5b                                              	pop    rbx
  402941:	5d                                              	pop    rbp
  402942:	41 5c                                           	pop    r12
  402944:	41 5d                                           	pop    r13
  402946:	41 5e                                           	pop    r14
  402948:	41 5f                                           	pop    r15
  40294a:	c3                                              	ret
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPcSt6vectorIcSaIcEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_algo.h:1962
  40294b:	90                                              	nop
  40294c:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

0000000000402950 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag.isra.0>:
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:206
  402950:	41 54                                           	push   r12
  402952:	49 89 d4                                        	mov    r12,rdx
  402955:	55                                              	push   rbp
  402956:	48 89 f5                                        	mov    rbp,rsi
  402959:	53                                              	push   rbx
  40295a:	48 89 fb                                        	mov    rbx,rdi
  40295d:	48 83 ec 10                                     	sub    rsp,0x10
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:211
  402961:	48 85 d2                                        	test   rdx,rdx
  402964:	74 05                                           	je     40296b <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag.isra.0+0x1b>
  402966:	48 85 f6                                        	test   rsi,rsi
  402969:	74 78                                           	je     4029e3 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag.isra.0+0x93>
_ZSt10__distanceIPcENSt15iterator_traitsIT_E15difference_typeES2_S2_St26random_access_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator_base_funcs.h:104
  40296b:	49 29 ec                                        	sub    r12,rbp
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:215
  40296e:	4c 89 64 24 08                                  	mov    QWORD PTR [rsp+0x8],r12
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:217
  402973:	49 83 fc 0f                                     	cmp    r12,0xf
  402977:	77 37                                           	ja     4029b0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag.isra.0+0x60>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402979:	48 8b 3b                                        	mov    rdi,QWORD PTR [rbx]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:348
  40297c:	49 83 fc 01                                     	cmp    r12,0x1
  402980:	75 26                                           	jne    4029a8 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag.isra.0+0x58>
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  402982:	0f b6 45 00                                     	movzx  eax,BYTE PTR [rbp+0x0]
  402986:	88 07                                           	mov    BYTE PTR [rdi],al
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:232
  402988:	4c 8b 64 24 08                                  	mov    r12,QWORD PTR [rsp+0x8]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  40298d:	48 8b 3b                                        	mov    rdi,QWORD PTR [rbx]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:183
  402990:	4c 89 63 08                                     	mov    QWORD PTR [rbx+0x8],r12
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  402994:	42 c6 04 27 00                                  	mov    BYTE PTR [rdi+r12*1],0x0
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:233
  402999:	48 83 c4 10                                     	add    rsp,0x10
  40299d:	5b                                              	pop    rbx
  40299e:	5d                                              	pop    rbp
  40299f:	41 5c                                           	pop    r12
  4029a1:	c3                                              	ret
  4029a2:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
_ZNSt11char_traitsIcE4copyEPcPKcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:389
  4029a8:	4d 85 e4                                        	test   r12,r12
  4029ab:	74 e3                                           	je     402990 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag.isra.0+0x40>
  4029ad:	eb 1f                                           	jmp    4029ce <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag.isra.0+0x7e>
  4029af:	90                                              	nop
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:219
  4029b0:	48 89 df                                        	mov    rdi,rbx
  4029b3:	31 d2                                           	xor    edx,edx
  4029b5:	48 8d 74 24 08                                  	lea    rsi,[rsp+0x8]
  4029ba:	e8 31 f9 ff ff                                  	call   4022f0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@plt>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:179
  4029bf:	48 89 03                                        	mov    QWORD PTR [rbx],rax
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:219
  4029c2:	48 89 c7                                        	mov    rdi,rax
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:211
  4029c5:	48 8b 44 24 08                                  	mov    rax,QWORD PTR [rsp+0x8]
  4029ca:	48 89 43 10                                     	mov    QWORD PTR [rbx+0x10],rax
_ZNSt11char_traitsIcE4copyEPcPKcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:395
  4029ce:	4c 89 e2                                        	mov    rdx,r12
  4029d1:	48 89 ee                                        	mov    rsi,rbp
  4029d4:	e8 a7 f7 ff ff                                  	call   402180 <memcpy@plt>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:232
  4029d9:	4c 8b 64 24 08                                  	mov    r12,QWORD PTR [rsp+0x8]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  4029de:	48 8b 3b                                        	mov    rdi,QWORD PTR [rbx]
_ZNSt11char_traitsIcE4copyEPcPKcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:395
  4029e1:	eb ad                                           	jmp    402990 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag.isra.0+0x40>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:212
  4029e3:	bf 08 40 41 00                                  	mov    edi,0x414008
  4029e8:	e8 73 f7 ff ff                                  	call   402160 <_ZSt19__throw_logic_errorPKc@plt>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag.isra.0():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:212
  4029ed:	90                                              	nop
  4029ee:	66 90                                           	xchg   ax,ax

00000000004029f0 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0>:
_ZNKSt7__cxx1112regex_traitsIcE5valueEci():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.tcc:338
  4029f0:	41 57                                           	push   r15
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.tcc:341
  4029f2:	40 0f be d7                                     	movsx  edx,dil
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.tcc:338
  4029f6:	41 56                                           	push   r14
  4029f8:	41 55                                           	push   r13
  4029fa:	41 54                                           	push   r12
  4029fc:	55                                              	push   rbp
  4029fd:	89 f5                                           	mov    ebp,esi
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC4EmcRKS3_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:542
  4029ff:	be 01 00 00 00                                  	mov    esi,0x1
_ZNKSt7__cxx1112regex_traitsIcE5valueEci():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.tcc:338
  402a04:	53                                              	push   rbx
  402a05:	48 81 ec b8 01 00 00                            	sub    rsp,0x1b8
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC4EmcRKS3_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:542
  402a0c:	48 8d 7c 24 10                                  	lea    rdi,[rsp+0x10]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC4EPcRKS3_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:157
  402a11:	4c 8d 7c 24 20                                  	lea    r15,[rsp+0x20]
  402a16:	4c 89 7c 24 10                                  	mov    QWORD PTR [rsp+0x10],r15
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC4EmcRKS3_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:542
  402a1b:	e8 10 f8 ff ff                                  	call   402230 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc@plt>
_ZNSt9basic_iosIcSt11char_traitsIcEEC4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_ios.h:462
  402a20:	48 8d bc 24 a8 00 00 00                         	lea    rdi,[rsp+0xa8]
  402a28:	e8 23 f6 ff ff                                  	call   402050 <_ZNSt8ios_baseC2Ev@plt>
_ZNSiC4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/istream:607
  402a2d:	48 8b 1d 3c 72 01 00                            	mov    rbx,QWORD PTR [rip+0x1723c]        # 419c70 <_ZTTNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEE@@GLIBCXX_3.4.21+0x8>
_ZNSt9basic_iosIcSt11char_traitsIcEEC4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_ios.h:462
  402a34:	31 c0                                           	xor    eax,eax
_ZNSiC4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/istream:608
  402a36:	31 f6                                           	xor    esi,esi
_ZNSt9basic_iosIcSt11char_traitsIcEEC4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_ios.h:462
  402a38:	66 0f ef c0                                     	pxor   xmm0,xmm0
  402a3c:	66 89 84 24 88 01 00 00                         	mov    WORD PTR [rsp+0x188],ax
_ZNSiC4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/istream:607
  402a44:	48 8b 0d 2d 72 01 00                            	mov    rcx,QWORD PTR [rip+0x1722d]        # 419c78 <_ZTTNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEE@@GLIBCXX_3.4.21+0x10>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/istream:608
  402a4b:	48 8d 7c 24 30                                  	lea    rdi,[rsp+0x30]
_ZNSt9basic_iosIcSt11char_traitsIcEEC4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_ios.h:462
  402a50:	0f 29 84 24 90 01 00 00                         	movaps XMMWORD PTR [rsp+0x190],xmm0
  402a58:	0f 29 84 24 a0 01 00 00                         	movaps XMMWORD PTR [rsp+0x1a0],xmm0
_ZNSiC4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/istream:607
  402a60:	48 8b 43 e8                                     	mov    rax,QWORD PTR [rbx-0x18]
_ZNSt9basic_iosIcSt11char_traitsIcEEC4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_ios.h:462
  402a64:	48 c7 84 24 a8 00 00 00 18 9b 41 00             	mov    QWORD PTR [rsp+0xa8],0x419b18
  402a70:	48 c7 84 24 80 01 00 00 00 00 00 00             	mov    QWORD PTR [rsp+0x180],0x0
_ZNSiC4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/istream:607
  402a7c:	48 89 5c 24 30                                  	mov    QWORD PTR [rsp+0x30],rbx
  402a81:	48 89 4c 04 30                                  	mov    QWORD PTR [rsp+rax*1+0x30],rcx
  402a86:	48 c7 44 24 38 00 00 00 00                      	mov    QWORD PTR [rsp+0x38],0x0
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/istream:608
  402a8f:	48 03 7b e8                                     	add    rdi,QWORD PTR [rbx-0x18]
  402a93:	e8 c8 f7 ff ff                                  	call   402260 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
_ZNSt15basic_streambufIcSt11char_traitsIcEEC4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/streambuf:473
  402a98:	48 8d 7c 24 78                                  	lea    rdi,[rsp+0x78]
_ZNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEC4ERKNS_12basic_stringIcS2_S3_EESt13_Ios_Openmode():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/sstream:462
  402a9d:	48 c7 44 24 30 40 9b 41 00                      	mov    QWORD PTR [rsp+0x30],0x419b40
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC4EPcRKS3_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:157
  402aa6:	4c 8d ac 24 98 00 00 00                         	lea    r13,[rsp+0x98]
_ZNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEC4ERKNS_12basic_stringIcS2_S3_EESt13_Ios_Openmode():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/sstream:462
  402aae:	48 c7 84 24 a8 00 00 00 68 9b 41 00             	mov    QWORD PTR [rsp+0xa8],0x419b68
_ZNSt15basic_streambufIcSt11char_traitsIcEEC4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/streambuf:473
  402aba:	48 c7 44 24 40 88 9b 41 00                      	mov    QWORD PTR [rsp+0x40],0x419b88
  402ac3:	48 c7 44 24 48 00 00 00 00                      	mov    QWORD PTR [rsp+0x48],0x0
  402acc:	48 c7 44 24 50 00 00 00 00                      	mov    QWORD PTR [rsp+0x50],0x0
  402ad5:	48 c7 44 24 58 00 00 00 00                      	mov    QWORD PTR [rsp+0x58],0x0
  402ade:	48 c7 44 24 60 00 00 00 00                      	mov    QWORD PTR [rsp+0x60],0x0
  402ae7:	48 c7 44 24 68 00 00 00 00                      	mov    QWORD PTR [rsp+0x68],0x0
  402af0:	48 c7 44 24 70 00 00 00 00                      	mov    QWORD PTR [rsp+0x70],0x0
  402af9:	e8 52 f8 ff ff                                  	call   402350 <_ZNSt6localeC1Ev@plt>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402afe:	4c 8b 74 24 10                                  	mov    r14,QWORD PTR [rsp+0x10]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:902
  402b03:	4c 8b 64 24 18                                  	mov    r12,QWORD PTR [rsp+0x18]
_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEC4ERKNS_12basic_stringIcS2_S3_EESt13_Ios_Openmode():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/sstream:127
  402b08:	48 c7 44 24 40 48 9d 41 00                      	mov    QWORD PTR [rsp+0x40],0x419d48
  402b11:	c7 84 24 80 00 00 00 00 00 00 00                	mov    DWORD PTR [rsp+0x80],0x0
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:211
  402b1c:	4c 89 f0                                        	mov    rax,r14
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC4EPcRKS3_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:157
  402b1f:	4c 89 ac 24 88 00 00 00                         	mov    QWORD PTR [rsp+0x88],r13
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:211
  402b27:	4c 01 e0                                        	add    rax,r12
  402b2a:	74 09                                           	je     402b35 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x145>
  402b2c:	4d 85 f6                                        	test   r14,r14
  402b2f:	0f 84 fa 01 00 00                               	je     402d2f <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x33f>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:215
  402b35:	4c 89 64 24 08                                  	mov    QWORD PTR [rsp+0x8],r12
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:217
  402b3a:	49 83 fc 0f                                     	cmp    r12,0xf
  402b3e:	0f 87 5c 01 00 00                               	ja     402ca0 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x2b0>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:348
  402b44:	49 83 fc 01                                     	cmp    r12,0x1
  402b48:	0f 85 3a 01 00 00                               	jne    402c88 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x298>
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  402b4e:	41 0f b6 06                                     	movzx  eax,BYTE PTR [r14]
  402b52:	88 84 24 98 00 00 00                            	mov    BYTE PTR [rsp+0x98],al
  402b59:	4c 89 e8                                        	mov    rax,r13
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:183
  402b5c:	4c 89 a4 24 90 00 00 00                         	mov    QWORD PTR [rsp+0x90],r12
_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE17_M_stringbuf_initESt13_Ios_Openmode():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/sstream:219
  402b64:	31 c9                                           	xor    ecx,ecx
  402b66:	31 d2                                           	xor    edx,edx
  402b68:	48 8d 7c 24 40                                  	lea    rdi,[rsp+0x40]
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  402b6d:	42 c6 04 20 00                                  	mov    BYTE PTR [rax+r12*1],0x0
_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE17_M_stringbuf_initESt13_Ios_Openmode():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/sstream:219
  402b72:	48 8b b4 24 88 00 00 00                         	mov    rsi,QWORD PTR [rsp+0x88]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/sstream:215
  402b7a:	c7 84 24 80 00 00 00 08 00 00 00                	mov    DWORD PTR [rsp+0x80],0x8
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/sstream:219
  402b85:	e8 c6 f6 ff ff                                  	call   402250 <_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE7_M_syncEPcmm@plt>
_ZNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEC4ERKNS_12basic_stringIcS2_S3_EESt13_Ios_Openmode():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/sstream:463
  402b8a:	48 8d 74 24 40                                  	lea    rsi,[rsp+0x40]
  402b8f:	48 8d bc 24 a8 00 00 00                         	lea    rdi,[rsp+0xa8]
  402b97:	e8 c4 f6 ff ff                                  	call   402260 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402b9c:	48 8b 7c 24 10                                  	mov    rdi,QWORD PTR [rsp+0x10]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:231
  402ba1:	4c 39 ff                                        	cmp    rdi,r15
  402ba4:	74 0e                                           	je     402bb4 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x1c4>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:237
  402ba6:	48 8b 44 24 20                                  	mov    rax,QWORD PTR [rsp+0x20]
  402bab:	48 8d 70 01                                     	lea    rsi,[rax+0x1]
_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  402baf:	e8 0c f6 ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNKSt7__cxx1112regex_traitsIcE5valueEci():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.tcc:343
  402bb4:	83 fd 08                                        	cmp    ebp,0x8
  402bb7:	0f 84 33 01 00 00                               	je     402cf0 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x300>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.tcc:345
  402bbd:	83 fd 10                                        	cmp    ebp,0x10
  402bc0:	0f 84 4a 01 00 00                               	je     402d10 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x320>
_ZNSirsERl():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/istream:187
  402bc6:	48 8d 74 24 10                                  	lea    rsi,[rsp+0x10]
  402bcb:	48 8d 7c 24 30                                  	lea    rdi,[rsp+0x30]
  402bd0:	e8 fb f5 ff ff                                  	call   4021d0 <_ZNSi10_M_extractIlEERSiRT_@plt>
_ZNKSt7__cxx1112regex_traitsIcE5valueEci():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.tcc:348
  402bd5:	f6 84 24 c8 00 00 00 05                         	test   BYTE PTR [rsp+0xc8],0x5
  402bdd:	0f 85 5d 01 00 00                               	jne    402d40 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x350>
  402be3:	44 8b 64 24 10                                  	mov    r12d,DWORD PTR [rsp+0x10]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402be8:	48 8b bc 24 88 00 00 00                         	mov    rdi,QWORD PTR [rsp+0x88]
_ZNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/sstream:472
  402bf0:	48 c7 44 24 30 40 9b 41 00                      	mov    QWORD PTR [rsp+0x30],0x419b40
  402bf9:	48 c7 84 24 a8 00 00 00 68 9b 41 00             	mov    QWORD PTR [rsp+0xa8],0x419b68
_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/sstream:65
  402c05:	48 c7 44 24 40 48 9d 41 00                      	mov    QWORD PTR [rsp+0x40],0x419d48
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:231
  402c0e:	4c 39 ef                                        	cmp    rdi,r13
  402c11:	74 11                                           	je     402c24 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x234>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:237
  402c13:	48 8b 84 24 98 00 00 00                         	mov    rax,QWORD PTR [rsp+0x98]
  402c1b:	48 8d 70 01                                     	lea    rsi,[rax+0x1]
_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  402c1f:	e8 9c f5 ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt15basic_streambufIcSt11char_traitsIcEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/streambuf:205
  402c24:	48 c7 44 24 40 88 9b 41 00                      	mov    QWORD PTR [rsp+0x40],0x419b88
  402c2d:	48 8d 7c 24 78                                  	lea    rdi,[rsp+0x78]
  402c32:	e8 49 f6 ff ff                                  	call   402280 <_ZNSt6localeD1Ev@plt>
_ZNSiD4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/istream:104
  402c37:	48 8b 43 e8                                     	mov    rax,QWORD PTR [rbx-0x18]
  402c3b:	48 89 5c 24 30                                  	mov    QWORD PTR [rsp+0x30],rbx
_ZNSt9basic_iosIcSt11char_traitsIcEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_ios.h:282
  402c40:	48 8d bc 24 a8 00 00 00                         	lea    rdi,[rsp+0xa8]
_ZNSiD4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/istream:104
  402c48:	48 8b 0d 29 70 01 00                            	mov    rcx,QWORD PTR [rip+0x17029]        # 419c78 <_ZTTNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEE@@GLIBCXX_3.4.21+0x10>
  402c4f:	48 89 4c 04 30                                  	mov    QWORD PTR [rsp+rax*1+0x30],rcx
  402c54:	48 c7 44 24 38 00 00 00 00                      	mov    QWORD PTR [rsp+0x38],0x0
_ZNSt9basic_iosIcSt11char_traitsIcEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_ios.h:282
  402c5d:	48 c7 84 24 a8 00 00 00 18 9b 41 00             	mov    QWORD PTR [rsp+0xa8],0x419b18
  402c69:	e8 02 f4 ff ff                                  	call   402070 <_ZNSt8ios_baseD2Ev@plt>
_ZNKSt7__cxx1112regex_traitsIcE5valueEci():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.tcc:349
  402c6e:	48 81 c4 b8 01 00 00                            	add    rsp,0x1b8
  402c75:	44 89 e0                                        	mov    eax,r12d
  402c78:	5b                                              	pop    rbx
  402c79:	5d                                              	pop    rbp
  402c7a:	41 5c                                           	pop    r12
  402c7c:	41 5d                                           	pop    r13
  402c7e:	41 5e                                           	pop    r14
  402c80:	41 5f                                           	pop    r15
  402c82:	c3                                              	ret
  402c83:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]
_ZNSt11char_traitsIcE4copyEPcPKcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:389
  402c88:	4d 85 e4                                        	test   r12,r12
  402c8b:	0f 85 ba 00 00 00                               	jne    402d4b <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x35b>
  402c91:	4c 89 e8                                        	mov    rax,r13
  402c94:	e9 c3 fe ff ff                                  	jmp    402b5c <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x16c>
  402c99:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:219
  402ca0:	31 d2                                           	xor    edx,edx
  402ca2:	48 8d 74 24 08                                  	lea    rsi,[rsp+0x8]
  402ca7:	48 8d bc 24 88 00 00 00                         	lea    rdi,[rsp+0x88]
  402caf:	e8 3c f6 ff ff                                  	call   4022f0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@plt>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:179
  402cb4:	48 89 84 24 88 00 00 00                         	mov    QWORD PTR [rsp+0x88],rax
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:219
  402cbc:	48 89 c7                                        	mov    rdi,rax
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:211
  402cbf:	48 8b 44 24 08                                  	mov    rax,QWORD PTR [rsp+0x8]
  402cc4:	48 89 84 24 98 00 00 00                         	mov    QWORD PTR [rsp+0x98],rax
_ZNSt11char_traitsIcE4copyEPcPKcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:395
  402ccc:	4c 89 e2                                        	mov    rdx,r12
  402ccf:	4c 89 f6                                        	mov    rsi,r14
  402cd2:	e8 a9 f4 ff ff                                  	call   402180 <memcpy@plt>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:232
  402cd7:	4c 8b 64 24 08                                  	mov    r12,QWORD PTR [rsp+0x8]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402cdc:	48 8b 84 24 88 00 00 00                         	mov    rax,QWORD PTR [rsp+0x88]
_ZNSt11char_traitsIcE4copyEPcPKcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:395
  402ce4:	e9 73 fe ff ff                                  	jmp    402b5c <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x16c>
  402ce9:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
_ZNSirsEPFRSt8ios_baseS0_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/istream:133
  402cf0:	48 8b 44 24 30                                  	mov    rax,QWORD PTR [rsp+0x30]
  402cf5:	48 8d 54 24 30                                  	lea    rdx,[rsp+0x30]
  402cfa:	48 03 50 e8                                     	add    rdx,QWORD PTR [rax-0x18]
_ZStanSt13_Ios_FmtflagsS_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/ios_base.h:84
  402cfe:	8b 42 18                                        	mov    eax,DWORD PTR [rdx+0x18]
  402d01:	83 e0 b5                                        	and    eax,0xffffffb5
_ZStorSt13_Ios_FmtflagsS_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/ios_base.h:88
  402d04:	83 c8 40                                        	or     eax,0x40
  402d07:	89 42 18                                        	mov    DWORD PTR [rdx+0x18],eax
_ZSt3octRSt8ios_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/ios_base.h:1042
  402d0a:	e9 b7 fe ff ff                                  	jmp    402bc6 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x1d6>
  402d0f:	90                                              	nop
_ZNSirsEPFRSt8ios_baseS0_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/istream:133
  402d10:	48 8b 44 24 30                                  	mov    rax,QWORD PTR [rsp+0x30]
  402d15:	48 8d 54 24 30                                  	lea    rdx,[rsp+0x30]
  402d1a:	48 03 50 e8                                     	add    rdx,QWORD PTR [rax-0x18]
_ZStanSt13_Ios_FmtflagsS_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/ios_base.h:84
  402d1e:	8b 42 18                                        	mov    eax,DWORD PTR [rdx+0x18]
  402d21:	83 e0 b5                                        	and    eax,0xffffffb5
_ZStorSt13_Ios_FmtflagsS_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/ios_base.h:88
  402d24:	83 c8 08                                        	or     eax,0x8
  402d27:	89 42 18                                        	mov    DWORD PTR [rdx+0x18],eax
_ZSt3hexRSt8ios_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/ios_base.h:1034
  402d2a:	e9 97 fe ff ff                                  	jmp    402bc6 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x1d6>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:212
  402d2f:	bf 08 40 41 00                                  	mov    edi,0x414008
  402d34:	e8 27 f4 ff ff                                  	call   402160 <_ZSt19__throw_logic_errorPKc@plt>
  402d39:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
_ZNKSt7__cxx1112regex_traitsIcE5valueEci():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.tcc:348
  402d40:	41 bc ff ff ff ff                               	mov    r12d,0xffffffff
  402d46:	e9 9d fe ff ff                                  	jmp    402be8 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x1f8>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402d4b:	4c 89 ef                                        	mov    rdi,r13
  402d4e:	e9 79 ff ff ff                                  	jmp    402ccc <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0+0x2dc>
_ZNSt15basic_streambufIcSt11char_traitsIcEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/streambuf:205
  402d53:	48 89 c5                                        	mov    rbp,rax
  402d56:	e9 80 f6 ff ff                                  	jmp    4023db <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0.cold+0x27>
_ZNSt9basic_iosIcSt11char_traitsIcEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_ios.h:282
  402d5b:	48 89 c5                                        	mov    rbp,rax
  402d5e:	e9 a9 f6 ff ff                                  	jmp    40240c <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0.cold+0x58>
_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/sstream:65
  402d63:	48 89 c5                                        	mov    rbp,rax
  402d66:	e9 49 f6 ff ff                                  	jmp    4023b4 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0.cold>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402d6b:	48 89 c5                                        	mov    rbp,rax
  402d6e:	e9 4a f6 ff ff                                  	jmp    4023bd <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0.cold+0x9>
_ZNKSt7__cxx1112regex_traitsIcE5valueEci():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.tcc:341
  402d73:	48 89 c5                                        	mov    rbp,rax
  402d76:	e9 ca f6 ff ff                                  	jmp    402445 <_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0.cold+0x91>
_ZNKSt7__cxx1112regex_traitsIcE5valueEci.isra.0():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.tcc:341
  402d7b:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]

0000000000402d80 <_Z9regexTestv>:
_Z9regexTestv():
/tmp/compiler-explorer-compiler202095-9137-1kkxb4c.yd7w/example.cpp:4
  402d80:	41 54                                           	push   r12
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:219
  402d82:	31 d2                                           	xor    edx,edx
_Z9regexTestv():
/tmp/compiler-explorer-compiler202095-9137-1kkxb4c.yd7w/example.cpp:4
  402d84:	55                                              	push   rbp
  402d85:	53                                              	push   rbx
  402d86:	48 81 ec e0 01 00 00                            	sub    rsp,0x1e0
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:219
  402d8d:	48 8d 74 24 50                                  	lea    rsi,[rsp+0x50]
  402d92:	48 8d 7c 24 10                                  	lea    rdi,[rsp+0x10]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:215
  402d97:	48 c7 44 24 50 77 00 00 00                      	mov    QWORD PTR [rsp+0x50],0x77
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC4EPcRKS3_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:157
  402da0:	4c 8d 64 24 20                                  	lea    r12,[rsp+0x20]
  402da5:	4c 89 64 24 10                                  	mov    QWORD PTR [rsp+0x10],r12
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:219
  402daa:	e8 41 f5 ff ff                                  	call   4022f0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@plt>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:211
  402daf:	48 8b 54 24 50                                  	mov    rdx,QWORD PTR [rsp+0x50]
_ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEEC4IPKcEET_S7_NSt15regex_constants18syntax_option_typeE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:507
  402db4:	48 8d 7c 24 08                                  	lea    rdi,[rsp+0x8]
_ZNSt11char_traitsIcE4copyEPcPKcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:395
  402db9:	66 0f 6f 05 1f 24 01 00                         	movdqa xmm0,XMMWORD PTR [rip+0x1241f]        # 4151e0 <_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag+0x30>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:179
  402dc1:	48 89 44 24 10                                  	mov    QWORD PTR [rsp+0x10],rax
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:211
  402dc6:	48 89 54 24 20                                  	mov    QWORD PTR [rsp+0x20],rdx
_ZNSt11char_traitsIcE4copyEPcPKcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:395
  402dcb:	ba 6d 73 00 00                                  	mov    edx,0x736d
  402dd0:	0f 11 00                                        	movups XMMWORD PTR [rax],xmm0
  402dd3:	66 0f 6f 05 15 24 01 00                         	movdqa xmm0,XMMWORD PTR [rip+0x12415]        # 4151f0 <_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag+0x40>
  402ddb:	66 89 50 74                                     	mov    WORD PTR [rax+0x74],dx
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  402ddf:	48 8b 54 24 10                                  	mov    rdx,QWORD PTR [rsp+0x10]
_ZNSt11char_traitsIcE4copyEPcPKcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:395
  402de4:	0f 11 40 10                                     	movups XMMWORD PTR [rax+0x10],xmm0
  402de8:	66 0f 6f 05 10 24 01 00                         	movdqa xmm0,XMMWORD PTR [rip+0x12410]        # 415200 <_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag+0x50>
  402df0:	c7 40 70 6f 62 6c 65                            	mov    DWORD PTR [rax+0x70],0x656c626f
  402df7:	0f 11 40 20                                     	movups XMMWORD PTR [rax+0x20],xmm0
  402dfb:	66 0f 6f 05 0d 24 01 00                         	movdqa xmm0,XMMWORD PTR [rip+0x1240d]        # 415210 <_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag+0x60>
  402e03:	c6 40 76 2e                                     	mov    BYTE PTR [rax+0x76],0x2e
  402e07:	0f 11 40 30                                     	movups XMMWORD PTR [rax+0x30],xmm0
  402e0b:	66 0f 6f 05 0d 24 01 00                         	movdqa xmm0,XMMWORD PTR [rip+0x1240d]        # 415220 <_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag+0x70>
  402e13:	0f 11 40 40                                     	movups XMMWORD PTR [rax+0x40],xmm0
  402e17:	66 0f 6f 05 11 24 01 00                         	movdqa xmm0,XMMWORD PTR [rip+0x12411]        # 415230 <_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag+0x80>
  402e1f:	0f 11 40 50                                     	movups XMMWORD PTR [rax+0x50],xmm0
  402e23:	66 0f 6f 05 15 24 01 00                         	movdqa xmm0,XMMWORD PTR [rip+0x12415]        # 415240 <_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag+0x90>
  402e2b:	0f 11 40 60                                     	movups XMMWORD PTR [rax+0x60],xmm0
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.tcc:232
  402e2f:	48 8b 44 24 50                                  	mov    rax,QWORD PTR [rsp+0x50]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:183
  402e34:	48 89 44 24 18                                  	mov    QWORD PTR [rsp+0x18],rax
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  402e39:	c6 04 02 00                                     	mov    BYTE PTR [rdx+rax*1],0x0
_ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEEC4IPKcEET_S7_NSt15regex_constants18syntax_option_typeE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:507
  402e3d:	e8 0e f5 ff ff                                  	call   402350 <_ZNSt6localeC1Ev@plt>
_ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEEC4IPKcEET_S7_St6localeNSt15regex_constants18syntax_option_typeE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:764
  402e42:	48 8d 74 24 08                                  	lea    rsi,[rsp+0x8]
  402e47:	48 8d 7c 24 38                                  	lea    rdi,[rsp+0x38]
  402e4c:	c7 44 24 30 11 00 00 00                         	mov    DWORD PTR [rsp+0x30],0x11
  402e54:	e8 c7 f2 ff ff                                  	call   402120 <_ZNSt6localeC1ERKS_@plt>
_ZNSt8__detail13__compile_nfaINSt7__cxx1112regex_traitsIcEEPKcEENSt9enable_ifIXsrNS_20__is_contiguous_iterIT0_EE5valueESt10shared_ptrIKNS_4_NFAIT_EEEE4typeES8_S8_RKNSC_11locale_typeENSt15regex_constants18syntax_option_typeE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:183
  402e59:	44 8b 44 24 30                                  	mov    r8d,DWORD PTR [rsp+0x30]
  402e5e:	48 8d 4c 24 38                                  	lea    rcx,[rsp+0x38]
  402e63:	ba 52 46 41 00                                  	mov    edx,0x414652
  402e68:	be 3f 46 41 00                                  	mov    esi,0x41463f
  402e6d:	48 8d 7c 24 50                                  	lea    rdi,[rsp+0x50]
  402e72:	e8 69 fc 00 00                                  	call   412ae0 <_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEEC1EPKcS6_RKSt6localeNSt15regex_constants18syntax_option_typeE>
_ZNSt12__shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EEC4IS5_vEEOS_IT_LS8_2EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:1199
  402e77:	66 0f 6f 8c 24 50 01 00 00                      	movdqa xmm1,XMMWORD PTR [rsp+0x150]
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:596
  402e80:	48 8b bc 24 80 01 00 00                         	mov    rdi,QWORD PTR [rsp+0x180]
_ZNSt12__shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EEC4IS5_vEEOS_IT_LS8_2EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:1202
  402e88:	66 0f ef c0                                     	pxor   xmm0,xmm0
  402e8c:	0f 29 84 24 50 01 00 00                         	movaps XMMWORD PTR [rsp+0x150],xmm0
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:1199
  402e94:	0f 29 4c 24 40                                  	movaps XMMWORD PTR [rsp+0x40],xmm1
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:596
  402e99:	48 85 ff                                        	test   rdi,rdi
  402e9c:	74 55                                           	je     402ef3 <_Z9regexTestv+0x173>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:598
  402e9e:	48 8b 84 24 c8 01 00 00                         	mov    rax,QWORD PTR [rsp+0x1c8]
  402ea6:	48 8b 9c 24 a8 01 00 00                         	mov    rbx,QWORD PTR [rsp+0x1a8]
  402eae:	48 8d 68 08                                     	lea    rbp,[rax+0x8]
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE16_M_destroy_nodesEPPS5_S9_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:675
  402eb2:	48 39 dd                                        	cmp    rbp,rbx
  402eb5:	76 27                                           	jbe    402ede <_Z9regexTestv+0x15e>
  402eb7:	66 0f 1f 84 00 00 00 00 00                      	nop    WORD PTR [rax+rax*1+0x0]
_ZN9__gnu_cxx13new_allocatorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEE10deallocateEPS6_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  402ec0:	48 8b 3b                                        	mov    rdi,QWORD PTR [rbx]
  402ec3:	be f8 01 00 00                                  	mov    esi,0x1f8
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE16_M_destroy_nodesEPPS5_S9_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:675
  402ec8:	48 83 c3 08                                     	add    rbx,0x8
_ZN9__gnu_cxx13new_allocatorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEE10deallocateEPS6_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  402ecc:	e8 ef f2 ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE16_M_destroy_nodesEPPS5_S9_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:675
  402ed1:	48 39 dd                                        	cmp    rbp,rbx
  402ed4:	77 ea                                           	ja     402ec0 <_Z9regexTestv+0x140>
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:600
  402ed6:	48 8b bc 24 80 01 00 00                         	mov    rdi,QWORD PTR [rsp+0x180]
_ZN9__gnu_cxx13new_allocatorIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEE10deallocateEPS7_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  402ede:	48 8b 84 24 88 01 00 00                         	mov    rax,QWORD PTR [rsp+0x188]
  402ee6:	48 8d 34 c5 00 00 00 00                         	lea    rsi,[rax*8+0x0]
  402eee:	e8 cd f2 ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402ef3:	48 8b bc 24 60 01 00 00                         	mov    rdi,QWORD PTR [rsp+0x160]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:231
  402efb:	48 8d 84 24 70 01 00 00                         	lea    rax,[rsp+0x170]
  402f03:	48 39 c7                                        	cmp    rdi,rax
  402f06:	74 11                                           	je     402f19 <_Z9regexTestv+0x199>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:237
  402f08:	48 8b 84 24 70 01 00 00                         	mov    rax,QWORD PTR [rsp+0x170]
  402f10:	48 8d 70 01                                     	lea    rsi,[rax+0x1]
_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  402f14:	e8 a7 f2 ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt14__shared_countILN9__gnu_cxx12_Lock_policyE2EED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:732
  402f19:	48 8b ac 24 58 01 00 00                         	mov    rbp,QWORD PTR [rsp+0x158]
  402f21:	48 85 ed                                        	test   rbp,rbp
  402f24:	74 20                                           	je     402f46 <_Z9regexTestv+0x1c6>
_ZN9__gnu_cxx27__exchange_and_add_dispatchEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:83
  402f26:	bb 00 00 00 00                                  	mov    ebx,0x0
  402f2b:	48 85 db                                        	test   rbx,rbx
  402f2e:	0f 85 fc 00 00 00                               	jne    403030 <_Z9regexTestv+0x2b0>
_ZN9__gnu_cxx25__exchange_and_add_singleEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:68
  402f34:	8b 45 08                                        	mov    eax,DWORD PTR [rbp+0x8]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:69
  402f37:	8d 50 ff                                        	lea    edx,[rax-0x1]
  402f3a:	89 55 08                                        	mov    DWORD PTR [rbp+0x8],edx
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:155
  402f3d:	83 f8 01                                        	cmp    eax,0x1
  402f40:	0f 84 8a 00 00 00                               	je     402fd0 <_Z9regexTestv+0x250>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402f46:	48 8b bc 24 20 01 00 00                         	mov    rdi,QWORD PTR [rsp+0x120]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:231
  402f4e:	48 8d 84 24 30 01 00 00                         	lea    rax,[rsp+0x130]
  402f56:	48 39 c7                                        	cmp    rdi,rax
  402f59:	74 11                                           	je     402f6c <_Z9regexTestv+0x1ec>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:237
  402f5b:	48 8b 84 24 30 01 00 00                         	mov    rax,QWORD PTR [rsp+0x130]
  402f63:	48 8d 70 01                                     	lea    rsi,[rax+0x1]
_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  402f67:	e8 54 f2 ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEEC4IPKcEET_S7_NSt15regex_constants18syntax_option_typeE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:507
  402f6c:	48 8d 7c 24 08                                  	lea    rdi,[rsp+0x8]
  402f71:	e8 0a f3 ff ff                                  	call   402280 <_ZNSt6localeD1Ev@plt>
_ZNSt14__shared_countILN9__gnu_cxx12_Lock_policyE2EED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:732
  402f76:	48 8b 6c 24 48                                  	mov    rbp,QWORD PTR [rsp+0x48]
  402f7b:	48 85 ed                                        	test   rbp,rbp
  402f7e:	74 1c                                           	je     402f9c <_Z9regexTestv+0x21c>
_ZN9__gnu_cxx27__exchange_and_add_dispatchEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:83
  402f80:	bb 00 00 00 00                                  	mov    ebx,0x0
  402f85:	48 85 db                                        	test   rbx,rbx
  402f88:	0f 85 b2 00 00 00                               	jne    403040 <_Z9regexTestv+0x2c0>
_ZN9__gnu_cxx25__exchange_and_add_singleEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:68
  402f8e:	8b 45 08                                        	mov    eax,DWORD PTR [rbp+0x8]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:69
  402f91:	8d 50 ff                                        	lea    edx,[rax-0x1]
  402f94:	89 55 08                                        	mov    DWORD PTR [rbp+0x8],edx
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:155
  402f97:	83 f8 01                                        	cmp    eax,0x1
  402f9a:	74 64                                           	je     403000 <_Z9regexTestv+0x280>
_ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:526
  402f9c:	48 8d 7c 24 38                                  	lea    rdi,[rsp+0x38]
  402fa1:	e8 da f2 ff ff                                  	call   402280 <_ZNSt6localeD1Ev@plt>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  402fa6:	48 8b 7c 24 10                                  	mov    rdi,QWORD PTR [rsp+0x10]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:231
  402fab:	4c 39 e7                                        	cmp    rdi,r12
  402fae:	74 0e                                           	je     402fbe <_Z9regexTestv+0x23e>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:237
  402fb0:	48 8b 44 24 20                                  	mov    rax,QWORD PTR [rsp+0x20]
  402fb5:	48 8d 70 01                                     	lea    rsi,[rax+0x1]
_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  402fb9:	e8 02 f2 ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_Z9regexTestv():
/tmp/compiler-explorer-compiler202095-9137-1kkxb4c.yd7w/example.cpp:10
  402fbe:	48 81 c4 e0 01 00 00                            	add    rsp,0x1e0
  402fc5:	5b                                              	pop    rbx
  402fc6:	5d                                              	pop    rbp
  402fc7:	41 5c                                           	pop    r12
  402fc9:	c3                                              	ret
  402fca:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:158
  402fd0:	48 8b 45 00                                     	mov    rax,QWORD PTR [rbp+0x0]
  402fd4:	48 89 ef                                        	mov    rdi,rbp
  402fd7:	ff 50 10                                        	call   QWORD PTR [rax+0x10]
_ZN9__gnu_cxx27__exchange_and_add_dispatchEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:83
  402fda:	48 85 db                                        	test   rbx,rbx
  402fdd:	75 7c                                           	jne    40305b <_Z9regexTestv+0x2db>
_ZN9__gnu_cxx25__exchange_and_add_singleEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:68
  402fdf:	8b 45 0c                                        	mov    eax,DWORD PTR [rbp+0xc]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:69
  402fe2:	8d 50 ff                                        	lea    edx,[rax-0x1]
  402fe5:	89 55 0c                                        	mov    DWORD PTR [rbp+0xc],edx
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:170
  402fe8:	83 f8 01                                        	cmp    eax,0x1
  402feb:	0f 85 55 ff ff ff                               	jne    402f46 <_Z9regexTestv+0x1c6>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:174
  402ff1:	48 8b 45 00                                     	mov    rax,QWORD PTR [rbp+0x0]
  402ff5:	48 89 ef                                        	mov    rdi,rbp
  402ff8:	ff 50 18                                        	call   QWORD PTR [rax+0x18]
  402ffb:	e9 46 ff ff ff                                  	jmp    402f46 <_Z9regexTestv+0x1c6>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:158
  403000:	48 8b 45 00                                     	mov    rax,QWORD PTR [rbp+0x0]
  403004:	48 89 ef                                        	mov    rdi,rbp
  403007:	ff 50 10                                        	call   QWORD PTR [rax+0x10]
_ZN9__gnu_cxx27__exchange_and_add_dispatchEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:83
  40300a:	48 85 db                                        	test   rbx,rbx
  40300d:	75 40                                           	jne    40304f <_Z9regexTestv+0x2cf>
_ZN9__gnu_cxx25__exchange_and_add_singleEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:68
  40300f:	8b 45 0c                                        	mov    eax,DWORD PTR [rbp+0xc]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:69
  403012:	8d 50 ff                                        	lea    edx,[rax-0x1]
  403015:	89 55 0c                                        	mov    DWORD PTR [rbp+0xc],edx
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:170
  403018:	83 f8 01                                        	cmp    eax,0x1
  40301b:	0f 85 7b ff ff ff                               	jne    402f9c <_Z9regexTestv+0x21c>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:174
  403021:	48 8b 45 00                                     	mov    rax,QWORD PTR [rbp+0x0]
  403025:	48 89 ef                                        	mov    rdi,rbp
  403028:	ff 50 18                                        	call   QWORD PTR [rax+0x18]
  40302b:	e9 6c ff ff ff                                  	jmp    402f9c <_Z9regexTestv+0x21c>
_ZN9__gnu_cxx18__exchange_and_addEPVii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:50
  403030:	b8 ff ff ff ff                                  	mov    eax,0xffffffff
  403035:	f0 0f c1 45 08                                  	lock xadd DWORD PTR [rbp+0x8],eax
_ZN9__gnu_cxx27__exchange_and_add_dispatchEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:84
  40303a:	e9 fe fe ff ff                                  	jmp    402f3d <_Z9regexTestv+0x1bd>
  40303f:	90                                              	nop
_ZN9__gnu_cxx18__exchange_and_addEPVii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:50
  403040:	b8 ff ff ff ff                                  	mov    eax,0xffffffff
  403045:	f0 0f c1 45 08                                  	lock xadd DWORD PTR [rbp+0x8],eax
_ZN9__gnu_cxx27__exchange_and_add_dispatchEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:84
  40304a:	e9 48 ff ff ff                                  	jmp    402f97 <_Z9regexTestv+0x217>
_ZN9__gnu_cxx18__exchange_and_addEPVii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:50
  40304f:	b8 ff ff ff ff                                  	mov    eax,0xffffffff
  403054:	f0 0f c1 45 0c                                  	lock xadd DWORD PTR [rbp+0xc],eax
_ZN9__gnu_cxx27__exchange_and_add_dispatchEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:84
  403059:	eb bd                                           	jmp    403018 <_Z9regexTestv+0x298>
_ZN9__gnu_cxx18__exchange_and_addEPVii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:50
  40305b:	b8 ff ff ff ff                                  	mov    eax,0xffffffff
  403060:	f0 0f c1 45 0c                                  	lock xadd DWORD PTR [rbp+0xc],eax
_ZN9__gnu_cxx27__exchange_and_add_dispatchEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:84
  403065:	e9 7e ff ff ff                                  	jmp    402fe8 <_Z9regexTestv+0x268>
_ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEEC4IPKcEET_S7_St6localeNSt15regex_constants18syntax_option_typeE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:764
  40306a:	48 89 c5                                        	mov    rbp,rax
  40306d:	e9 e5 f3 ff ff                                  	jmp    402457 <_Z9regexTestv.cold>
_Z9regexTestv():
  403072:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  40307c:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

0000000000403080 <_ZNKSt5ctypeIcE8do_widenEc>:
_ZNKSt5ctypeIcE8do_widenEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:1084
  403080:	89 f0                                           	mov    eax,esi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:1085
  403082:	c3                                              	ret
  403083:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  40308d:	0f 1f 00                                        	nop    DWORD PTR [rax]

0000000000403090 <_ZNKSt5ctypeIcE9do_narrowEcc>:
_ZNKSt5ctypeIcE9do_narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:1134
  403090:	89 f0                                           	mov    eax,esi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:1135
  403092:	c3                                              	ret
  403093:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  40309d:	0f 1f 00                                        	nop    DWORD PTR [rax]

00000000004030a0 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZSt13__invoke_implIbRNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb0EEEJcEET_St14__invoke_otherOT0_DpOT1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/invoke.h:60
  4030a0:	0f b6 16                                        	movzx  edx,BYTE PTR [rsi]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb0EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:376
  4030a3:	80 fa 0d                                        	cmp    dl,0xd
  4030a6:	0f 95 c0                                        	setne  al
  4030a9:	80 fa 0a                                        	cmp    dl,0xa
  4030ac:	0f 95 c2                                        	setne  dl
  4030af:	21 d0                                           	and    eax,edx
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  4030b1:	c3                                              	ret
  4030b2:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  4030bc:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

00000000004030c0 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZSt13__invoke_implIbRNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEJcEET_St14__invoke_otherOT0_DpOT1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/invoke.h:60
  4030c0:	0f b6 16                                        	movzx  edx,BYTE PTR [rsi]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:376
  4030c3:	80 fa 0d                                        	cmp    dl,0xd
  4030c6:	0f 95 c0                                        	setne  al
  4030c9:	80 fa 0a                                        	cmp    dl,0xa
  4030cc:	0f 95 c2                                        	setne  dl
  4030cf:	21 d0                                           	and    eax,edx
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  4030d1:	c3                                              	ret
  4030d2:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  4030dc:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

00000000004030e0 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNKSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:405
  4030e0:	0f b6 06                                        	movzx  eax,BYTE PTR [rsi]
  4030e3:	38 47 01                                        	cmp    BYTE PTR [rdi+0x1],al
  4030e6:	0f 94 c0                                        	sete   al
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  4030e9:	c3                                              	ret
  4030ea:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]

00000000004030f0 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNKSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:405
  4030f0:	0f b6 06                                        	movzx  eax,BYTE PTR [rsi]
  4030f3:	38 47 08                                        	cmp    BYTE PTR [rdi+0x8],al
  4030f6:	0f 94 c0                                        	sete   al
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  4030f9:	c3                                              	ret
  4030fa:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]

0000000000403100 <_ZNSt17_Function_handlerIFbcENSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNKSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:527
  403100:	0f b6 16                                        	movzx  edx,BYTE PTR [rsi]
_ZNKSt12_Base_bitsetILm4EE10_M_getwordEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:120
  403103:	48 8b 37                                        	mov    rsi,QWORD PTR [rdi]
_ZNSt12_Base_bitsetILm4EE10_S_maskbitEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:112
  403106:	b8 01 00 00 00                                  	mov    eax,0x1
_ZNKSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:527
  40310b:	48 89 d1                                        	mov    rcx,rdx
_ZNSt12_Base_bitsetILm4EE12_S_whichwordEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:100
  40310e:	48 c1 ea 06                                     	shr    rdx,0x6
_ZNSt12_Base_bitsetILm4EE10_S_maskbitEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:112
  403112:	48 d3 e0                                        	shl    rax,cl
_ZNKSt6bitsetILm256EE15_Unchecked_testEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:1066
  403115:	48 23 44 d6 78                                  	and    rax,QWORD PTR [rsi+rdx*8+0x78]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:1067
  40311a:	0f 95 c0                                        	setne  al
_ZNSt17_Function_handlerIFbcENSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  40311d:	c3                                              	ret
  40311e:	66 90                                           	xchg   ax,ax

0000000000403120 <_ZNSt17_Function_handlerIFbcENSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNKSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:527
  403120:	0f b6 16                                        	movzx  edx,BYTE PTR [rsi]
_ZNKSt12_Base_bitsetILm4EE10_M_getwordEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:120
  403123:	48 8b 37                                        	mov    rsi,QWORD PTR [rdi]
_ZNSt12_Base_bitsetILm4EE10_S_maskbitEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:112
  403126:	b8 01 00 00 00                                  	mov    eax,0x1
_ZNKSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:527
  40312b:	48 89 d1                                        	mov    rcx,rdx
_ZNSt12_Base_bitsetILm4EE12_S_whichwordEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:100
  40312e:	48 c1 ea 06                                     	shr    rdx,0x6
_ZNSt12_Base_bitsetILm4EE10_S_maskbitEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:112
  403132:	48 d3 e0                                        	shl    rax,cl
_ZNKSt6bitsetILm256EE15_Unchecked_testEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:1066
  403135:	48 23 84 d6 80 00 00 00                         	and    rax,QWORD PTR [rsi+rdx*8+0x80]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:1067
  40313d:	0f 95 c0                                        	setne  al
_ZNSt17_Function_handlerIFbcENSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403140:	c3                                              	ret
  403141:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  40314b:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]

0000000000403150 <_ZNSt17_Function_handlerIFbcENSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNKSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:527
  403150:	0f b6 16                                        	movzx  edx,BYTE PTR [rsi]
_ZNKSt12_Base_bitsetILm4EE10_M_getwordEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:120
  403153:	48 8b 37                                        	mov    rsi,QWORD PTR [rdi]
_ZNSt12_Base_bitsetILm4EE10_S_maskbitEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:112
  403156:	b8 01 00 00 00                                  	mov    eax,0x1
_ZNKSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:527
  40315b:	48 89 d1                                        	mov    rcx,rdx
_ZNSt12_Base_bitsetILm4EE12_S_whichwordEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:100
  40315e:	48 c1 ea 06                                     	shr    rdx,0x6
_ZNSt12_Base_bitsetILm4EE10_S_maskbitEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:112
  403162:	48 d3 e0                                        	shl    rax,cl
_ZNKSt6bitsetILm256EE15_Unchecked_testEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:1066
  403165:	48 23 84 d6 80 00 00 00                         	and    rax,QWORD PTR [rsi+rdx*8+0x80]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:1067
  40316d:	0f 95 c0                                        	setne  al
_ZNSt17_Function_handlerIFbcENSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403170:	c3                                              	ret
  403171:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  40317b:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]

0000000000403180 <_ZNSt17_Function_handlerIFbcENSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNKSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:527
  403180:	0f b6 16                                        	movzx  edx,BYTE PTR [rsi]
_ZNKSt12_Base_bitsetILm4EE10_M_getwordEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:120
  403183:	48 8b 37                                        	mov    rsi,QWORD PTR [rdi]
_ZNSt12_Base_bitsetILm4EE10_S_maskbitEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:112
  403186:	b8 01 00 00 00                                  	mov    eax,0x1
_ZNKSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:527
  40318b:	48 89 d1                                        	mov    rcx,rdx
_ZNSt12_Base_bitsetILm4EE12_S_whichwordEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:100
  40318e:	48 c1 ea 06                                     	shr    rdx,0x6
_ZNSt12_Base_bitsetILm4EE10_S_maskbitEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:112
  403192:	48 d3 e0                                        	shl    rax,cl
_ZNKSt6bitsetILm256EE15_Unchecked_testEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:1066
  403195:	48 23 84 d6 80 00 00 00                         	and    rax,QWORD PTR [rsi+rdx*8+0x80]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bitset:1067
  40319d:	0f 95 c0                                        	setne  al
_ZNSt17_Function_handlerIFbcENSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  4031a0:	c3                                              	ret
  4031a1:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  4031ab:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]

00000000004031b0 <_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EED1Ev>:
_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EED1Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:555
  4031b0:	c3                                              	ret
  4031b1:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  4031bb:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]

00000000004031c0 <_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EED0Ev>:
_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EED0Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:555
  4031c0:	be 68 00 00 00                                  	mov    esi,0x68
  4031c5:	e9 f6 ef ff ff                                  	jmp    4021c0 <_ZdlPvm@plt>
  4031ca:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]

00000000004031d0 <_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE10_M_destroyEv>:
_ZN9__gnu_cxx13new_allocatorISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS7_ELNS_12_Lock_policyE2EEE10deallocateEPSA_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  4031d0:	be 68 00 00 00                                  	mov    esi,0x68
  4031d5:	e9 e6 ef ff ff                                  	jmp    4021c0 <_ZdlPvm@plt>
_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE10_M_destroyEv():
  4031da:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]

00000000004031e0 <_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info>:
_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:578
  4031e0:	41 54                                           	push   r12
_ZN9__gnu_cxx16__aligned_bufferINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEE7_M_addrEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/aligned_buffer.h:104
  4031e2:	4c 8d 67 10                                     	lea    r12,[rdi+0x10]
_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:586
  4031e6:	48 81 fe b0 51 41 00                            	cmp    rsi,0x4151b0
  4031ed:	74 27                                           	je     403216 <_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info+0x36>
_ZNKSt9type_infoeqERKS_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/typeinfo:122
  4031ef:	48 8b 7e 08                                     	mov    rdi,QWORD PTR [rsi+0x8]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/typeinfo:123
  4031f3:	48 81 ff 60 49 41 00                            	cmp    rdi,0x414960
  4031fa:	74 1a                                           	je     403216 <_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info+0x36>
  4031fc:	80 3f 2a                                        	cmp    BYTE PTR [rdi],0x2a
  4031ff:	74 1f                                           	je     403220 <_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info+0x40>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/typeinfo:124
  403201:	be 60 49 41 00                                  	mov    esi,0x414960
  403206:	e8 65 f0 ff ff                                  	call   402270 <strcmp@plt>
_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:594
  40320b:	85 c0                                           	test   eax,eax
  40320d:	b8 00 00 00 00                                  	mov    eax,0x0
  403212:	4c 0f 45 e0                                     	cmovne r12,rax
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:595
  403216:	4c 89 e0                                        	mov    rax,r12
  403219:	41 5c                                           	pop    r12
  40321b:	c3                                              	ret
  40321c:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:594
  403220:	45 31 e4                                        	xor    r12d,r12d
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:595
  403223:	4c 89 e0                                        	mov    rax,r12
  403226:	41 5c                                           	pop    r12
  403228:	c3                                              	ret
  403229:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]

0000000000403230 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:271
  403230:	85 d2                                           	test   edx,edx
  403232:	74 0c                                           	je     403240 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x10>
  403234:	83 fa 01                                        	cmp    edx,0x1
  403237:	75 03                                           	jne    40323c <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0xc>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:279
  403239:	48 89 37                                        	mov    QWORD PTR [rdi],rsi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40323c:	31 c0                                           	xor    eax,eax
  40323e:	c3                                              	ret
  40323f:	90                                              	nop
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:275
  403240:	48 c7 07 c8 4a 41 00                            	mov    QWORD PTR [rdi],0x414ac8
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  403247:	31 c0                                           	xor    eax,eax
  403249:	c3                                              	ret
  40324a:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]

0000000000403250 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:271
  403250:	85 d2                                           	test   edx,edx
  403252:	74 14                                           	je     403268 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x18>
  403254:	83 fa 01                                        	cmp    edx,0x1
  403257:	74 1f                                           	je     403278 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x28>
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:184
  403259:	83 fa 02                                        	cmp    edx,0x2
  40325c:	74 22                                           	je     403280 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x30>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40325e:	31 c0                                           	xor    eax,eax
  403260:	c3                                              	ret
  403261:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:275
  403268:	48 c7 07 28 4b 41 00                            	mov    QWORD PTR [rdi],0x414b28
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40326f:	31 c0                                           	xor    eax,eax
  403271:	c3                                              	ret
  403272:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:279
  403278:	48 89 37                                        	mov    QWORD PTR [rdi],rsi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40327b:	31 c0                                           	xor    eax,eax
  40327d:	c3                                              	ret
  40327e:	66 90                                           	xchg   ax,ax
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE8_M_cloneERSt9_Any_dataRKS8_St17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/new:175
  403280:	48 8b 06                                        	mov    rax,QWORD PTR [rsi]
  403283:	48 89 07                                        	mov    QWORD PTR [rdi],rax
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:197
  403286:	eb d6                                           	jmp    40325e <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0xe>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
  403288:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]

0000000000403290 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:271
  403290:	85 d2                                           	test   edx,edx
  403292:	74 14                                           	je     4032a8 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x18>
  403294:	83 fa 01                                        	cmp    edx,0x1
  403297:	74 1f                                           	je     4032b8 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x28>
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:184
  403299:	83 fa 02                                        	cmp    edx,0x2
  40329c:	74 22                                           	je     4032c0 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x30>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40329e:	31 c0                                           	xor    eax,eax
  4032a0:	c3                                              	ret
  4032a1:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:275
  4032a8:	48 c7 07 88 4b 41 00                            	mov    QWORD PTR [rdi],0x414b88
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  4032af:	31 c0                                           	xor    eax,eax
  4032b1:	c3                                              	ret
  4032b2:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:279
  4032b8:	48 89 37                                        	mov    QWORD PTR [rdi],rsi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  4032bb:	31 c0                                           	xor    eax,eax
  4032bd:	c3                                              	ret
  4032be:	66 90                                           	xchg   ax,ax
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE8_M_cloneERSt9_Any_dataRKS8_St17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/new:175
  4032c0:	48 8b 06                                        	mov    rax,QWORD PTR [rsi]
  4032c3:	48 89 07                                        	mov    QWORD PTR [rdi],rax
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:197
  4032c6:	eb d6                                           	jmp    40329e <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0xe>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
  4032c8:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]

00000000004032d0 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:271
  4032d0:	85 d2                                           	test   edx,edx
  4032d2:	74 14                                           	je     4032e8 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x18>
  4032d4:	83 fa 01                                        	cmp    edx,0x1
  4032d7:	74 1f                                           	je     4032f8 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x28>
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:184
  4032d9:	83 fa 02                                        	cmp    edx,0x2
  4032dc:	74 22                                           	je     403300 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x30>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  4032de:	31 c0                                           	xor    eax,eax
  4032e0:	c3                                              	ret
  4032e1:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:275
  4032e8:	48 c7 07 e8 4b 41 00                            	mov    QWORD PTR [rdi],0x414be8
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  4032ef:	31 c0                                           	xor    eax,eax
  4032f1:	c3                                              	ret
  4032f2:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:279
  4032f8:	48 89 37                                        	mov    QWORD PTR [rdi],rsi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  4032fb:	31 c0                                           	xor    eax,eax
  4032fd:	c3                                              	ret
  4032fe:	66 90                                           	xchg   ax,ax
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE8_M_cloneERSt9_Any_dataRKS8_St17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/new:175
  403300:	48 8b 06                                        	mov    rax,QWORD PTR [rsi]
  403303:	48 89 07                                        	mov    QWORD PTR [rdi],rax
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:197
  403306:	eb d6                                           	jmp    4032de <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0xe>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
  403308:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]

0000000000403310 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:271
  403310:	85 d2                                           	test   edx,edx
  403312:	74 0c                                           	je     403320 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x10>
  403314:	83 fa 01                                        	cmp    edx,0x1
  403317:	75 03                                           	jne    40331c <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0xc>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:279
  403319:	48 89 37                                        	mov    QWORD PTR [rdi],rsi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40331c:	31 c0                                           	xor    eax,eax
  40331e:	c3                                              	ret
  40331f:	90                                              	nop
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:275
  403320:	48 c7 07 48 4c 41 00                            	mov    QWORD PTR [rdi],0x414c48
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  403327:	31 c0                                           	xor    eax,eax
  403329:	c3                                              	ret
  40332a:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]

0000000000403330 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:271
  403330:	85 d2                                           	test   edx,edx
  403332:	74 14                                           	je     403348 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x18>
  403334:	83 fa 01                                        	cmp    edx,0x1
  403337:	74 1f                                           	je     403358 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x28>
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:184
  403339:	83 fa 02                                        	cmp    edx,0x2
  40333c:	74 22                                           	je     403360 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x30>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40333e:	31 c0                                           	xor    eax,eax
  403340:	c3                                              	ret
  403341:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:275
  403348:	48 c7 07 a8 4c 41 00                            	mov    QWORD PTR [rdi],0x414ca8
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40334f:	31 c0                                           	xor    eax,eax
  403351:	c3                                              	ret
  403352:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:279
  403358:	48 89 37                                        	mov    QWORD PTR [rdi],rsi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40335b:	31 c0                                           	xor    eax,eax
  40335d:	c3                                              	ret
  40335e:	66 90                                           	xchg   ax,ax
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEE8_M_cloneERSt9_Any_dataRKS8_St17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/new:175
  403360:	48 8b 06                                        	mov    rax,QWORD PTR [rsi]
  403363:	48 89 07                                        	mov    QWORD PTR [rdi],rax
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:197
  403366:	eb d6                                           	jmp    40333e <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0xe>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
  403368:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]

0000000000403370 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:271
  403370:	85 d2                                           	test   edx,edx
  403372:	74 14                                           	je     403388 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x18>
  403374:	83 fa 01                                        	cmp    edx,0x1
  403377:	74 1f                                           	je     403398 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x28>
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:184
  403379:	83 fa 02                                        	cmp    edx,0x2
  40337c:	74 22                                           	je     4033a0 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x30>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40337e:	31 c0                                           	xor    eax,eax
  403380:	c3                                              	ret
  403381:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:275
  403388:	48 c7 07 08 4d 41 00                            	mov    QWORD PTR [rdi],0x414d08
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40338f:	31 c0                                           	xor    eax,eax
  403391:	c3                                              	ret
  403392:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:279
  403398:	48 89 37                                        	mov    QWORD PTR [rdi],rsi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40339b:	31 c0                                           	xor    eax,eax
  40339d:	c3                                              	ret
  40339e:	66 90                                           	xchg   ax,ax
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE8_M_cloneERSt9_Any_dataRKS8_St17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/new:175
  4033a0:	48 8b 06                                        	mov    rax,QWORD PTR [rsi]
  4033a3:	48 89 07                                        	mov    QWORD PTR [rdi],rax
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:197
  4033a6:	eb d6                                           	jmp    40337e <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0xe>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
  4033a8:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]

00000000004033b0 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:271
  4033b0:	85 d2                                           	test   edx,edx
  4033b2:	74 14                                           	je     4033c8 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x18>
  4033b4:	83 fa 01                                        	cmp    edx,0x1
  4033b7:	74 1f                                           	je     4033d8 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x28>
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:184
  4033b9:	83 fa 02                                        	cmp    edx,0x2
  4033bc:	74 22                                           	je     4033e0 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x30>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  4033be:	31 c0                                           	xor    eax,eax
  4033c0:	c3                                              	ret
  4033c1:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:275
  4033c8:	48 c7 07 68 4d 41 00                            	mov    QWORD PTR [rdi],0x414d68
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  4033cf:	31 c0                                           	xor    eax,eax
  4033d1:	c3                                              	ret
  4033d2:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:279
  4033d8:	48 89 37                                        	mov    QWORD PTR [rdi],rsi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  4033db:	31 c0                                           	xor    eax,eax
  4033dd:	c3                                              	ret
  4033de:	66 90                                           	xchg   ax,ax
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE8_M_cloneERSt9_Any_dataRKS8_St17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/new:175
  4033e0:	48 8b 06                                        	mov    rax,QWORD PTR [rsi]
  4033e3:	48 89 07                                        	mov    QWORD PTR [rdi],rax
_ZNSt14_Function_base13_Base_managerINSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:197
  4033e6:	eb d6                                           	jmp    4033be <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0xe>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
  4033e8:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]

00000000004033f0 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation>:
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:271
  4033f0:	85 d2                                           	test   edx,edx
  4033f2:	74 14                                           	je     403408 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x18>
  4033f4:	83 fa 01                                        	cmp    edx,0x1
  4033f7:	74 1f                                           	je     403418 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x28>
_ZNSt14_Function_base13_Base_managerINSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:184
  4033f9:	83 fa 02                                        	cmp    edx,0x2
  4033fc:	74 22                                           	je     403420 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x30>
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  4033fe:	31 c0                                           	xor    eax,eax
  403400:	c3                                              	ret
  403401:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:275
  403408:	48 c7 07 c8 4d 41 00                            	mov    QWORD PTR [rdi],0x414dc8
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40340f:	31 c0                                           	xor    eax,eax
  403411:	c3                                              	ret
  403412:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:279
  403418:	48 89 37                                        	mov    QWORD PTR [rdi],rsi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40341b:	31 c0                                           	xor    eax,eax
  40341d:	c3                                              	ret
  40341e:	66 90                                           	xchg   ax,ax
_ZNSt14_Function_base13_Base_managerINSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE8_M_cloneERSt9_Any_dataRKS8_St17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/new:175
  403420:	0f b7 06                                        	movzx  eax,WORD PTR [rsi]
  403423:	66 89 07                                        	mov    WORD PTR [rdi],ax
_ZNSt14_Function_base13_Base_managerINSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:197
  403426:	eb d6                                           	jmp    4033fe <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0xe>
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
  403428:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]

0000000000403430 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation>:
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:271
  403430:	85 d2                                           	test   edx,edx
  403432:	74 14                                           	je     403448 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x18>
  403434:	83 fa 01                                        	cmp    edx,0x1
  403437:	74 1f                                           	je     403458 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x28>
_ZNSt14_Function_base13_Base_managerINSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:184
  403439:	83 fa 02                                        	cmp    edx,0x2
  40343c:	74 22                                           	je     403460 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x30>
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40343e:	31 c0                                           	xor    eax,eax
  403440:	c3                                              	ret
  403441:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:275
  403448:	48 c7 07 28 4e 41 00                            	mov    QWORD PTR [rdi],0x414e28
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40344f:	31 c0                                           	xor    eax,eax
  403451:	c3                                              	ret
  403452:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:279
  403458:	48 89 37                                        	mov    QWORD PTR [rdi],rsi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40345b:	31 c0                                           	xor    eax,eax
  40345d:	c3                                              	ret
  40345e:	66 90                                           	xchg   ax,ax
_ZNSt14_Function_base13_Base_managerINSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE8_M_cloneERSt9_Any_dataRKS8_St17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/new:175
  403460:	f3 0f 6f 06                                     	movdqu xmm0,XMMWORD PTR [rsi]
  403464:	0f 11 07                                        	movups XMMWORD PTR [rdi],xmm0
_ZNSt14_Function_base13_Base_managerINSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:197
  403467:	eb d5                                           	jmp    40343e <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0xe>
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
  403469:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]

0000000000403470 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation>:
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:271
  403470:	85 d2                                           	test   edx,edx
  403472:	74 14                                           	je     403488 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x18>
  403474:	83 fa 01                                        	cmp    edx,0x1
  403477:	74 1f                                           	je     403498 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x28>
_ZNSt14_Function_base13_Base_managerINSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:184
  403479:	83 fa 02                                        	cmp    edx,0x2
  40347c:	74 22                                           	je     4034a0 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x30>
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40347e:	31 c0                                           	xor    eax,eax
  403480:	c3                                              	ret
  403481:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:275
  403488:	48 c7 07 88 4e 41 00                            	mov    QWORD PTR [rdi],0x414e88
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40348f:	31 c0                                           	xor    eax,eax
  403491:	c3                                              	ret
  403492:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:279
  403498:	48 89 37                                        	mov    QWORD PTR [rdi],rsi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  40349b:	31 c0                                           	xor    eax,eax
  40349d:	c3                                              	ret
  40349e:	66 90                                           	xchg   ax,ax
_ZNSt14_Function_base13_Base_managerINSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE8_M_cloneERSt9_Any_dataRKS8_St17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/new:175
  4034a0:	f3 0f 6f 06                                     	movdqu xmm0,XMMWORD PTR [rsi]
  4034a4:	0f 11 07                                        	movups XMMWORD PTR [rdi],xmm0
_ZNSt14_Function_base13_Base_managerINSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:197
  4034a7:	eb d5                                           	jmp    40347e <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0xe>
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
  4034a9:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]

00000000004034b0 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation>:
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:271
  4034b0:	85 d2                                           	test   edx,edx
  4034b2:	74 14                                           	je     4034c8 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x18>
  4034b4:	83 fa 01                                        	cmp    edx,0x1
  4034b7:	74 1f                                           	je     4034d8 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x28>
_ZNSt14_Function_base13_Base_managerINSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:184
  4034b9:	83 fa 02                                        	cmp    edx,0x2
  4034bc:	74 22                                           	je     4034e0 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0x30>
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  4034be:	31 c0                                           	xor    eax,eax
  4034c0:	c3                                              	ret
  4034c1:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:275
  4034c8:	48 c7 07 e8 4e 41 00                            	mov    QWORD PTR [rdi],0x414ee8
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  4034cf:	31 c0                                           	xor    eax,eax
  4034d1:	c3                                              	ret
  4034d2:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:279
  4034d8:	48 89 37                                        	mov    QWORD PTR [rdi],rsi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:286
  4034db:	31 c0                                           	xor    eax,eax
  4034dd:	c3                                              	ret
  4034de:	66 90                                           	xchg   ax,ax
_ZNSt14_Function_base13_Base_managerINSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE8_M_cloneERSt9_Any_dataRKS8_St17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/new:175
  4034e0:	f3 0f 6f 06                                     	movdqu xmm0,XMMWORD PTR [rsi]
  4034e4:	0f 11 07                                        	movups XMMWORD PTR [rdi],xmm0
_ZNSt14_Function_base13_Base_managerINSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:197
  4034e7:	eb d5                                           	jmp    4034be <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation+0xe>
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation():
  4034e9:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]

00000000004034f0 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:289
  4034f0:	53                                              	push   rbx
_ZSt13__invoke_implIbRNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEJcEET_St14__invoke_otherOT0_DpOT1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/invoke.h:60
  4034f1:	0f b6 1e                                        	movzx  ebx,BYTE PTR [rsi]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:348
  4034f4:	0f b6 05 15 71 01 00                            	movzx  eax,BYTE PTR [rip+0x17115]        # 41a610 <_ZGVZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEclEcE5__nul>
  4034fb:	84 c0                                           	test   al,al
  4034fd:	74 11                                           	je     403510 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc+0x20>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:349
  4034ff:	38 1d 13 71 01 00                               	cmp    BYTE PTR [rip+0x17113],bl        # 41a618 <_ZZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEclEcE5__nul>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403505:	5b                                              	pop    rbx
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:349
  403506:	0f 95 c0                                        	setne  al
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403509:	c3                                              	ret
  40350a:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:348
  403510:	bf 10 a6 41 00                                  	mov    edi,0x41a610
  403515:	e8 f6 ed ff ff                                  	call   402310 <__cxa_guard_acquire@plt>
  40351a:	85 c0                                           	test   eax,eax
  40351c:	74 e1                                           	je     4034ff <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc+0xf>
  40351e:	bf 10 a6 41 00                                  	mov    edi,0x41a610
  403523:	c6 05 ee 70 01 00 00                            	mov    BYTE PTR [rip+0x170ee],0x0        # 41a618 <_ZZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEclEcE5__nul>
  40352a:	e8 11 ec ff ff                                  	call   402140 <__cxa_guard_release@plt>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:349
  40352f:	38 1d e3 70 01 00                               	cmp    BYTE PTR [rip+0x170e3],bl        # 41a618 <_ZZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEclEcE5__nul>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403535:	5b                                              	pop    rbx
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:349
  403536:	0f 95 c0                                        	setne  al
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403539:	c3                                              	ret
  40353a:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]

0000000000403540 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:289
  403540:	53                                              	push   rbx
_ZSt13__invoke_implIbRNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEJcEET_St14__invoke_otherOT0_DpOT1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/invoke.h:60
  403541:	0f b6 1e                                        	movzx  ebx,BYTE PTR [rsi]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:348
  403544:	0f b6 05 d5 70 01 00                            	movzx  eax,BYTE PTR [rip+0x170d5]        # 41a620 <_ZGVZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEclEcE5__nul>
  40354b:	84 c0                                           	test   al,al
  40354d:	74 11                                           	je     403560 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc+0x20>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:349
  40354f:	38 1d d3 70 01 00                               	cmp    BYTE PTR [rip+0x170d3],bl        # 41a628 <_ZZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEclEcE5__nul>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403555:	5b                                              	pop    rbx
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:349
  403556:	0f 95 c0                                        	setne  al
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403559:	c3                                              	ret
  40355a:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:348
  403560:	bf 20 a6 41 00                                  	mov    edi,0x41a620
  403565:	e8 a6 ed ff ff                                  	call   402310 <__cxa_guard_acquire@plt>
  40356a:	85 c0                                           	test   eax,eax
  40356c:	74 e1                                           	je     40354f <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc+0xf>
  40356e:	bf 20 a6 41 00                                  	mov    edi,0x41a620
  403573:	c6 05 ae 70 01 00 00                            	mov    BYTE PTR [rip+0x170ae],0x0        # 41a628 <_ZZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEclEcE5__nul>
  40357a:	e8 c1 eb ff ff                                  	call   402140 <__cxa_guard_release@plt>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:349
  40357f:	38 1d a3 70 01 00                               	cmp    BYTE PTR [rip+0x170a3],bl        # 41a628 <_ZZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEclEcE5__nul>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403585:	5b                                              	pop    rbx
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:349
  403586:	0f 95 c0                                        	setne  al
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403589:	c3                                              	ret
  40358a:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]

0000000000403590 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:289
  403590:	55                                              	push   rbp
  403591:	53                                              	push   rbx
  403592:	48 83 ec 08                                     	sub    rsp,0x8
_ZNKSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:405
  403596:	0f b6 5f 08                                     	movzx  ebx,BYTE PTR [rdi+0x8]
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  40359a:	48 8b 3f                                        	mov    rdi,QWORD PTR [rdi]
_ZSt13__invoke_implIbRNSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEJcEET_St14__invoke_otherOT0_DpOT1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/invoke.h:60
  40359d:	0f be 2e                                        	movsx  ebp,BYTE PTR [rsi]
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4035a0:	e8 0b eb ff ff                                  	call   4020b0 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@plt>
  4035a5:	48 89 c7                                        	mov    rdi,rax
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  4035a8:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
  4035ab:	89 ee                                           	mov    esi,ebp
  4035ad:	ff 50 20                                        	call   QWORD PTR [rax+0x20]
_ZNKSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:405
  4035b0:	38 c3                                           	cmp    bl,al
  4035b2:	0f 94 c0                                        	sete   al
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  4035b5:	48 83 c4 08                                     	add    rsp,0x8
  4035b9:	5b                                              	pop    rbx
  4035ba:	5d                                              	pop    rbp
  4035bb:	c3                                              	ret
  4035bc:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

00000000004035c0 <_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:289
  4035c0:	55                                              	push   rbp
  4035c1:	53                                              	push   rbx
  4035c2:	48 83 ec 08                                     	sub    rsp,0x8
_ZNKSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:405
  4035c6:	0f b6 5f 08                                     	movzx  ebx,BYTE PTR [rdi+0x8]
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4035ca:	48 8b 3f                                        	mov    rdi,QWORD PTR [rdi]
_ZSt13__invoke_implIbRNSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEJcEET_St14__invoke_otherOT0_DpOT1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/invoke.h:60
  4035cd:	0f be 2e                                        	movsx  ebp,BYTE PTR [rsi]
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4035d0:	e8 db ea ff ff                                  	call   4020b0 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@plt>
  4035d5:	48 89 c7                                        	mov    rdi,rax
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  4035d8:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
  4035db:	89 ee                                           	mov    esi,ebp
  4035dd:	ff 50 20                                        	call   QWORD PTR [rax+0x20]
_ZNKSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:405
  4035e0:	38 c3                                           	cmp    bl,al
  4035e2:	0f 94 c0                                        	sete   al
_ZNSt17_Function_handlerIFbcENSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  4035e5:	48 83 c4 08                                     	add    rsp,0x8
  4035e9:	5b                                              	pop    rbx
  4035ea:	5d                                              	pop    rbp
  4035eb:	c3                                              	ret
  4035ec:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

00000000004035f0 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0>:
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1913
  4035f0:	41 57                                           	push   r15
  4035f2:	41 56                                           	push   r14
  4035f4:	41 55                                           	push   r13
  4035f6:	41 54                                           	push   r12
  4035f8:	55                                              	push   rbp
  4035f9:	53                                              	push   rbx
  4035fa:	48 83 ec 28                                     	sub    rsp,0x28
  4035fe:	48 89 7c 24 10                                  	mov    QWORD PTR [rsp+0x10],rdi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  403603:	48 85 ff                                        	test   rdi,rdi
  403606:	0f 84 9b 01 00 00                               	je     4037a7 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x1b7>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_S_rightEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:798
  40360c:	48 8b 44 24 10                                  	mov    rax,QWORD PTR [rsp+0x10]
  403611:	4c 8b 70 18                                     	mov    r14,QWORD PTR [rax+0x18]
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  403615:	4d 85 f6                                        	test   r14,r14
  403618:	0f 84 67 01 00 00                               	je     403785 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x195>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_S_rightEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:798
  40361e:	4d 8b 7e 18                                     	mov    r15,QWORD PTR [r14+0x18]
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  403622:	4d 85 ff                                        	test   r15,r15
  403625:	0f 84 3c 01 00 00                               	je     403767 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x177>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_S_rightEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:798
  40362b:	49 8b 47 18                                     	mov    rax,QWORD PTR [r15+0x18]
  40362f:	48 89 44 24 08                                  	mov    QWORD PTR [rsp+0x8],rax
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  403634:	48 85 c0                                        	test   rax,rax
  403637:	0f 84 0c 01 00 00                               	je     403749 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x159>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_S_rightEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:798
  40363d:	48 8b 44 24 08                                  	mov    rax,QWORD PTR [rsp+0x8]
  403642:	48 8b 68 18                                     	mov    rbp,QWORD PTR [rax+0x18]
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  403646:	48 85 ed                                        	test   rbp,rbp
  403649:	0f 84 af 00 00 00                               	je     4036fe <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x10e>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_S_rightEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:798
  40364f:	4c 8b 6d 18                                     	mov    r13,QWORD PTR [rbp+0x18]
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  403653:	4d 85 ed                                        	test   r13,r13
  403656:	74 64                                           	je     4036bc <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0xcc>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_S_rightEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:798
  403658:	4d 8b 65 18                                     	mov    r12,QWORD PTR [r13+0x18]
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  40365c:	4d 85 e4                                        	test   r12,r12
  40365f:	74 7f                                           	je     4036e0 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0xf0>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_S_rightEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:798
  403661:	4d 8b 4c 24 18                                  	mov    r9,QWORD PTR [r12+0x18]
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  403666:	4d 85 c9                                        	test   r9,r9
  403669:	0f 84 b1 00 00 00                               	je     403720 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x130>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_S_rightEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:798
  40366f:	49 8b 59 18                                     	mov    rbx,QWORD PTR [r9+0x18]
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  403673:	48 85 db                                        	test   rbx,rbx
  403676:	74 29                                           	je     4036a1 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0xb1>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1919
  403678:	48 8b 7b 18                                     	mov    rdi,QWORD PTR [rbx+0x18]
  40367c:	4c 89 4c 24 18                                  	mov    QWORD PTR [rsp+0x18],r9
  403681:	e8 6a ff ff ff                                  	call   4035f0 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE7_S_leftEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:789
  403686:	48 89 df                                        	mov    rdi,rbx
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:790
  403689:	48 8b 5b 10                                     	mov    rbx,QWORD PTR [rbx+0x10]
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKllEEE10deallocateEPS5_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  40368d:	be 30 00 00 00                                  	mov    esi,0x30
  403692:	e8 29 eb ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  403697:	4c 8b 4c 24 18                                  	mov    r9,QWORD PTR [rsp+0x18]
  40369c:	48 85 db                                        	test   rbx,rbx
  40369f:	75 d7                                           	jne    403678 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x88>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE7_S_leftEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:790
  4036a1:	49 8b 59 10                                     	mov    rbx,QWORD PTR [r9+0x10]
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKllEEE10deallocateEPS5_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  4036a5:	be 30 00 00 00                                  	mov    esi,0x30
  4036aa:	4c 89 cf                                        	mov    rdi,r9
  4036ad:	e8 0e eb ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  4036b2:	48 85 db                                        	test   rbx,rbx
  4036b5:	74 69                                           	je     403720 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x130>
  4036b7:	49 89 d9                                        	mov    r9,rbx
  4036ba:	eb b3                                           	jmp    40366f <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x7f>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE7_S_leftEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:790
  4036bc:	4c 8b 65 10                                     	mov    r12,QWORD PTR [rbp+0x10]
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKllEEE10deallocateEPS5_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  4036c0:	be 30 00 00 00                                  	mov    esi,0x30
  4036c5:	48 89 ef                                        	mov    rdi,rbp
  4036c8:	e8 f3 ea ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  4036cd:	4d 85 e4                                        	test   r12,r12
  4036d0:	74 2c                                           	je     4036fe <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x10e>
  4036d2:	4c 89 e5                                        	mov    rbp,r12
  4036d5:	e9 75 ff ff ff                                  	jmp    40364f <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x5f>
  4036da:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE7_S_leftEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:790
  4036e0:	4d 8b 65 10                                     	mov    r12,QWORD PTR [r13+0x10]
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKllEEE10deallocateEPS5_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  4036e4:	be 30 00 00 00                                  	mov    esi,0x30
  4036e9:	4c 89 ef                                        	mov    rdi,r13
  4036ec:	e8 cf ea ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  4036f1:	4d 85 e4                                        	test   r12,r12
  4036f4:	74 c6                                           	je     4036bc <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0xcc>
  4036f6:	4d 89 e5                                        	mov    r13,r12
  4036f9:	e9 5a ff ff ff                                  	jmp    403658 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x68>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE7_S_leftEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:790
  4036fe:	48 8b 7c 24 08                                  	mov    rdi,QWORD PTR [rsp+0x8]
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKllEEE10deallocateEPS5_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  403703:	be 30 00 00 00                                  	mov    esi,0x30
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE7_S_leftEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:790
  403708:	48 8b 6f 10                                     	mov    rbp,QWORD PTR [rdi+0x10]
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKllEEE10deallocateEPS5_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  40370c:	e8 af ea ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  403711:	48 85 ed                                        	test   rbp,rbp
  403714:	74 33                                           	je     403749 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x159>
  403716:	48 89 6c 24 08                                  	mov    QWORD PTR [rsp+0x8],rbp
  40371b:	e9 1d ff ff ff                                  	jmp    40363d <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x4d>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE7_S_leftEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:790
  403720:	49 8b 44 24 10                                  	mov    rax,QWORD PTR [r12+0x10]
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKllEEE10deallocateEPS5_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  403725:	be 30 00 00 00                                  	mov    esi,0x30
  40372a:	4c 89 e7                                        	mov    rdi,r12
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE7_S_leftEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:790
  40372d:	48 89 44 24 18                                  	mov    QWORD PTR [rsp+0x18],rax
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKllEEE10deallocateEPS5_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  403732:	e8 89 ea ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  403737:	48 8b 44 24 18                                  	mov    rax,QWORD PTR [rsp+0x18]
  40373c:	48 85 c0                                        	test   rax,rax
  40373f:	74 9f                                           	je     4036e0 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0xf0>
  403741:	49 89 c4                                        	mov    r12,rax
  403744:	e9 18 ff ff ff                                  	jmp    403661 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x71>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE7_S_leftEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:790
  403749:	49 8b 5f 10                                     	mov    rbx,QWORD PTR [r15+0x10]
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKllEEE10deallocateEPS5_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  40374d:	be 30 00 00 00                                  	mov    esi,0x30
  403752:	4c 89 ff                                        	mov    rdi,r15
  403755:	e8 66 ea ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  40375a:	48 85 db                                        	test   rbx,rbx
  40375d:	74 08                                           	je     403767 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x177>
  40375f:	49 89 df                                        	mov    r15,rbx
  403762:	e9 c4 fe ff ff                                  	jmp    40362b <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x3b>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE7_S_leftEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:790
  403767:	49 8b 5e 10                                     	mov    rbx,QWORD PTR [r14+0x10]
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKllEEE10deallocateEPS5_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  40376b:	be 30 00 00 00                                  	mov    esi,0x30
  403770:	4c 89 f7                                        	mov    rdi,r14
  403773:	e8 48 ea ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  403778:	48 85 db                                        	test   rbx,rbx
  40377b:	74 08                                           	je     403785 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x195>
  40377d:	49 89 de                                        	mov    r14,rbx
  403780:	e9 99 fe ff ff                                  	jmp    40361e <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x2e>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE7_S_leftEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:790
  403785:	48 8b 7c 24 10                                  	mov    rdi,QWORD PTR [rsp+0x10]
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKllEEE10deallocateEPS5_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  40378a:	be 30 00 00 00                                  	mov    esi,0x30
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE7_S_leftEPSt18_Rb_tree_node_base():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:790
  40378f:	48 8b 5f 10                                     	mov    rbx,QWORD PTR [rdi+0x10]
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKllEEE10deallocateEPS5_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  403793:	e8 28 ea ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1917
  403798:	48 85 db                                        	test   rbx,rbx
  40379b:	74 0a                                           	je     4037a7 <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x1b7>
  40379d:	48 89 5c 24 10                                  	mov    QWORD PTR [rsp+0x10],rbx
  4037a2:	e9 65 fe ff ff                                  	jmp    40360c <_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0+0x1c>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_tree.h:1924
  4037a7:	48 83 c4 28                                     	add    rsp,0x28
  4037ab:	5b                                              	pop    rbx
  4037ac:	5d                                              	pop    rbp
  4037ad:	41 5c                                           	pop    r12
  4037af:	41 5d                                           	pop    r13
  4037b1:	41 5e                                           	pop    r14
  4037b3:	41 5f                                           	pop    r15
  4037b5:	c3                                              	ret
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE8_M_eraseEPSt13_Rb_tree_nodeIS2_E.isra.0():
  4037b6:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]

00000000004037c0 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:289
  4037c0:	41 54                                           	push   r12
  4037c2:	55                                              	push   rbp
  4037c3:	53                                              	push   rbx
  4037c4:	48 89 fb                                        	mov    rbx,rdi
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4037c7:	48 8b 3f                                        	mov    rdi,QWORD PTR [rdi]
_ZSt13__invoke_implIbRNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEJcEET_St14__invoke_otherOT0_DpOT1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/invoke.h:60
  4037ca:	0f be 2e                                        	movsx  ebp,BYTE PTR [rsi]
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4037cd:	e8 de e8 ff ff                                  	call   4020b0 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@plt>
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  4037d2:	89 ee                                           	mov    esi,ebp
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4037d4:	48 89 c7                                        	mov    rdi,rax
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  4037d7:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
  4037da:	ff 50 20                                        	call   QWORD PTR [rax+0x20]
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4037dd:	48 8b 3b                                        	mov    rdi,QWORD PTR [rbx]
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  4037e0:	89 c5                                           	mov    ebp,eax
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4037e2:	e8 c9 e8 ff ff                                  	call   4020b0 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@plt>
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  4037e7:	be 0a 00 00 00                                  	mov    esi,0xa
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4037ec:	48 89 c7                                        	mov    rdi,rax
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  4037ef:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
  4037f2:	ff 50 20                                        	call   QWORD PTR [rax+0x20]
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4037f5:	48 8b 3b                                        	mov    rdi,QWORD PTR [rbx]
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  4037f8:	41 89 c4                                        	mov    r12d,eax
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4037fb:	e8 b0 e8 ff ff                                  	call   4020b0 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@plt>
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  403800:	be 0d 00 00 00                                  	mov    esi,0xd
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  403805:	48 89 c7                                        	mov    rdi,rax
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  403808:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
  40380b:	ff 50 20                                        	call   QWORD PTR [rax+0x20]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:376
  40380e:	41 38 ec                                        	cmp    r12b,bpl
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403811:	5b                                              	pop    rbx
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  403812:	41 89 c0                                        	mov    r8d,eax
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:376
  403815:	0f 95 c0                                        	setne  al
  403818:	41 38 e8                                        	cmp    r8b,bpl
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  40381b:	5d                                              	pop    rbp
  40381c:	41 5c                                           	pop    r12
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:376
  40381e:	0f 95 c2                                        	setne  dl
  403821:	21 d0                                           	and    eax,edx
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403823:	c3                                              	ret
  403824:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  40382e:	66 90                                           	xchg   ax,ax

0000000000403830 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:289
  403830:	41 54                                           	push   r12
  403832:	55                                              	push   rbp
  403833:	53                                              	push   rbx
  403834:	48 89 fb                                        	mov    rbx,rdi
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  403837:	48 8b 3f                                        	mov    rdi,QWORD PTR [rdi]
_ZSt13__invoke_implIbRNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEJcEET_St14__invoke_otherOT0_DpOT1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/invoke.h:60
  40383a:	0f be 2e                                        	movsx  ebp,BYTE PTR [rsi]
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  40383d:	e8 6e e8 ff ff                                  	call   4020b0 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@plt>
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  403842:	89 ee                                           	mov    esi,ebp
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  403844:	48 89 c7                                        	mov    rdi,rax
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  403847:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
  40384a:	ff 50 20                                        	call   QWORD PTR [rax+0x20]
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  40384d:	48 8b 3b                                        	mov    rdi,QWORD PTR [rbx]
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  403850:	89 c5                                           	mov    ebp,eax
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  403852:	e8 59 e8 ff ff                                  	call   4020b0 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@plt>
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  403857:	be 0a 00 00 00                                  	mov    esi,0xa
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  40385c:	48 89 c7                                        	mov    rdi,rax
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  40385f:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
  403862:	ff 50 20                                        	call   QWORD PTR [rax+0x20]
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  403865:	48 8b 3b                                        	mov    rdi,QWORD PTR [rbx]
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  403868:	41 89 c4                                        	mov    r12d,eax
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  40386b:	e8 40 e8 ff ff                                  	call   4020b0 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@plt>
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  403870:	be 0d 00 00 00                                  	mov    esi,0xd
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  403875:	48 89 c7                                        	mov    rdi,rax
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  403878:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
  40387b:	ff 50 20                                        	call   QWORD PTR [rax+0x20]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:376
  40387e:	41 38 ec                                        	cmp    r12b,bpl
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403881:	5b                                              	pop    rbx
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  403882:	41 89 c0                                        	mov    r8d,eax
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:376
  403885:	0f 95 c0                                        	setne  al
  403888:	41 38 e8                                        	cmp    r8b,bpl
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  40388b:	5d                                              	pop    rbp
  40388c:	41 5c                                           	pop    r12
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EE8_M_applyEcSt17integral_constantIbLb1EE():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:376
  40388e:	0f 95 c2                                        	setne  dl
  403891:	21 d0                                           	and    eax,edx
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403893:	c3                                              	ret
  403894:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  40389e:	66 90                                           	xchg   ax,ax

00000000004038a0 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:289
  4038a0:	55                                              	push   rbp
  4038a1:	53                                              	push   rbx
  4038a2:	48 89 fb                                        	mov    rbx,rdi
  4038a5:	48 83 ec 08                                     	sub    rsp,0x8
_ZSt13__invoke_implIbRNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEJcEET_St14__invoke_otherOT0_DpOT1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/invoke.h:60
  4038a9:	0f be 2e                                        	movsx  ebp,BYTE PTR [rsi]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:348
  4038ac:	0f b6 05 3d 6d 01 00                            	movzx  eax,BYTE PTR [rip+0x16d3d]        # 41a5f0 <_ZGVZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEclEcE5__nul>
  4038b3:	84 c0                                           	test   al,al
  4038b5:	75 0e                                           	jne    4038c5 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc+0x25>
  4038b7:	bf f0 a5 41 00                                  	mov    edi,0x41a5f0
  4038bc:	e8 4f ea ff ff                                  	call   402310 <__cxa_guard_acquire@plt>
  4038c1:	85 c0                                           	test   eax,eax
  4038c3:	75 2b                                           	jne    4038f0 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc+0x50>
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4038c5:	48 8b 3b                                        	mov    rdi,QWORD PTR [rbx]
  4038c8:	e8 e3 e7 ff ff                                  	call   4020b0 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@plt>
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  4038cd:	89 ee                                           	mov    esi,ebp
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4038cf:	48 89 c7                                        	mov    rdi,rax
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  4038d2:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
  4038d5:	ff 50 20                                        	call   QWORD PTR [rax+0x20]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:349
  4038d8:	38 05 1a 6d 01 00                               	cmp    BYTE PTR [rip+0x16d1a],al        # 41a5f8 <_ZZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEclEcE5__nul>
  4038de:	0f 95 c0                                        	setne  al
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  4038e1:	48 83 c4 08                                     	add    rsp,0x8
  4038e5:	5b                                              	pop    rbx
  4038e6:	5d                                              	pop    rbp
  4038e7:	c3                                              	ret
  4038e8:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  4038f0:	48 8b 3b                                        	mov    rdi,QWORD PTR [rbx]
  4038f3:	e8 b8 e7 ff ff                                  	call   4020b0 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@plt>
  4038f8:	48 89 c7                                        	mov    rdi,rax
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  4038fb:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
  4038fe:	31 f6                                           	xor    esi,esi
  403900:	ff 50 20                                        	call   QWORD PTR [rax+0x20]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:348
  403903:	bf f0 a5 41 00                                  	mov    edi,0x41a5f0
  403908:	88 05 ea 6c 01 00                               	mov    BYTE PTR [rip+0x16cea],al        # 41a5f8 <_ZZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEclEcE5__nul>
  40390e:	e8 2d e8 ff ff                                  	call   402140 <__cxa_guard_release@plt>
  403913:	eb b0                                           	jmp    4038c5 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc+0x25>
  403915:	48 89 c5                                        	mov    rbp,rax
  403918:	bf f0 a5 41 00                                  	mov    edi,0x41a5f0
  40391d:	e8 0e e8 ff ff                                  	call   402130 <__cxa_guard_abort@plt>
  403922:	48 89 ef                                        	mov    rdi,rbp
  403925:	e8 b6 e9 ff ff                                  	call   4022e0 <_Unwind_Resume@plt>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEE9_M_invokeERKSt9_Any_dataOc():
  40392a:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]

0000000000403930 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc>:
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:289
  403930:	55                                              	push   rbp
  403931:	53                                              	push   rbx
  403932:	48 89 fb                                        	mov    rbx,rdi
  403935:	48 83 ec 08                                     	sub    rsp,0x8
_ZSt13__invoke_implIbRNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEJcEET_St14__invoke_otherOT0_DpOT1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/invoke.h:60
  403939:	0f be 2e                                        	movsx  ebp,BYTE PTR [rsi]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:348
  40393c:	0f b6 05 bd 6c 01 00                            	movzx  eax,BYTE PTR [rip+0x16cbd]        # 41a600 <_ZGVZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEclEcE5__nul>
  403943:	84 c0                                           	test   al,al
  403945:	75 0e                                           	jne    403955 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc+0x25>
  403947:	bf 00 a6 41 00                                  	mov    edi,0x41a600
  40394c:	e8 bf e9 ff ff                                  	call   402310 <__cxa_guard_acquire@plt>
  403951:	85 c0                                           	test   eax,eax
  403953:	75 2b                                           	jne    403980 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc+0x50>
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  403955:	48 8b 3b                                        	mov    rdi,QWORD PTR [rbx]
  403958:	e8 53 e7 ff ff                                  	call   4020b0 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@plt>
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  40395d:	89 ee                                           	mov    esi,ebp
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  40395f:	48 89 c7                                        	mov    rdi,rax
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  403962:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
  403965:	ff 50 20                                        	call   QWORD PTR [rax+0x20]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:349
  403968:	38 05 9a 6c 01 00                               	cmp    BYTE PTR [rip+0x16c9a],al        # 41a608 <_ZZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEclEcE5__nul>
  40396e:	0f 95 c0                                        	setne  al
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:293
  403971:	48 83 c4 08                                     	add    rsp,0x8
  403975:	5b                                              	pop    rbx
  403976:	5d                                              	pop    rbp
  403977:	c3                                              	ret
  403978:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]
_ZNKSt7__cxx1112regex_traitsIcE16translate_nocaseEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:199
  403980:	48 8b 3b                                        	mov    rdi,QWORD PTR [rbx]
  403983:	e8 28 e7 ff ff                                  	call   4020b0 <_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale@plt>
  403988:	48 89 c7                                        	mov    rdi,rax
_ZNKSt5ctypeIcE7tolowerEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:836
  40398b:	48 8b 00                                        	mov    rax,QWORD PTR [rax]
  40398e:	31 f6                                           	xor    esi,esi
  403990:	ff 50 20                                        	call   QWORD PTR [rax+0x20]
_ZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEclEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:348
  403993:	bf 00 a6 41 00                                  	mov    edi,0x41a600
  403998:	88 05 6a 6c 01 00                               	mov    BYTE PTR [rip+0x16c6a],al        # 41a608 <_ZZNKSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEclEcE5__nul>
  40399e:	e8 9d e7 ff ff                                  	call   402140 <__cxa_guard_release@plt>
  4039a3:	eb b0                                           	jmp    403955 <_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc+0x25>
  4039a5:	48 89 c5                                        	mov    rbp,rax
  4039a8:	bf 00 a6 41 00                                  	mov    edi,0x41a600
  4039ad:	e8 7e e7 ff ff                                  	call   402130 <__cxa_guard_abort@plt>
  4039b2:	48 89 ef                                        	mov    rdi,rbp
  4039b5:	e8 26 e9 ff ff                                  	call   4022e0 <_Unwind_Resume@plt>
_ZNSt17_Function_handlerIFbcENSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEE9_M_invokeERKSt9_Any_dataOc():
  4039ba:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]

00000000004039c0 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv>:
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:310
  4039c0:	41 56                                           	push   r14
  4039c2:	41 55                                           	push   r13
  4039c4:	41 54                                           	push   r12
  4039c6:	55                                              	push   rbp
  4039c7:	53                                              	push   rbx
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:313
  4039c8:	48 8b 87 b0 00 00 00                            	mov    rax,QWORD PTR [rdi+0xb0]
  4039cf:	48 3b 87 b8 00 00 00                            	cmp    rax,QWORD PTR [rdi+0xb8]
  4039d6:	0f 84 32 03 00 00                               	je     403d0e <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x34e>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:317
  4039dc:	48 8d 50 01                                     	lea    rdx,[rax+0x1]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:318
  4039e0:	4c 8b a7 c0 00 00 00                            	mov    r12,QWORD PTR [rdi+0xc0]
  4039e7:	48 89 fb                                        	mov    rbx,rdi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:317
  4039ea:	48 89 97 b0 00 00 00                            	mov    QWORD PTR [rdi+0xb0],rdx
  4039f1:	44 0f be 30                                     	movsx  r14d,BYTE PTR [rax]
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:933
  4039f5:	45 0f b6 ee                                     	movzx  r13d,r14b
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:317
  4039f9:	44 89 f5                                        	mov    ebp,r14d
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:933
  4039fc:	43 0f b6 8c 2c 39 01 00 00                      	movzx  ecx,BYTE PTR [r12+r13*1+0x139]
  403a05:	84 c9                                           	test   cl,cl
  403a07:	75 23                                           	jne    403a2c <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x6c>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:935
  403a09:	49 8b 04 24                                     	mov    rax,QWORD PTR [r12]
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:317
  403a0d:	44 89 f1                                        	mov    ecx,r14d
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:935
  403a10:	48 8b 40 40                                     	mov    rax,QWORD PTR [rax+0x40]
  403a14:	48 3d 90 30 40 00                               	cmp    rax,0x403090
  403a1a:	0f 85 c0 01 00 00                               	jne    403be0 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x220>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:936
  403a20:	84 c9                                           	test   cl,cl
  403a22:	74 08                                           	je     403a2c <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x6c>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:937
  403a24:	43 88 8c 2c 39 01 00 00                         	mov    BYTE PTR [r12+r13*1+0x139],cl
_ZNSt8__detail12_ScannerBase14_M_find_escapeEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.h:117
  403a2c:	48 8b 93 98 00 00 00                            	mov    rdx,QWORD PTR [rbx+0x98]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.h:118
  403a33:	0f b6 02                                        	movzx  eax,BYTE PTR [rdx]
  403a36:	84 c0                                           	test   al,al
  403a38:	75 16                                           	jne    403a50 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x90>
  403a3a:	e9 91 00 00 00                                  	jmp    403ad0 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x110>
  403a3f:	90                                              	nop
  403a40:	0f b6 42 02                                     	movzx  eax,BYTE PTR [rdx+0x2]
  403a44:	48 83 c2 02                                     	add    rdx,0x2
  403a48:	84 c0                                           	test   al,al
  403a4a:	0f 84 80 00 00 00                               	je     403ad0 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x110>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.h:119
  403a50:	38 c1                                           	cmp    cl,al
  403a52:	75 ec                                           	jne    403a40 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x80>
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:320
  403a54:	40 80 fd 62                                     	cmp    bpl,0x62
  403a58:	75 3e                                           	jne    403a98 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0xd8>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:320 (discriminator 2)
  403a5a:	83 bb 88 00 00 00 02                            	cmp    DWORD PTR [rbx+0x88],0x2
  403a61:	74 35                                           	je     403a98 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0xd8>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403a63:	48 8b 93 d0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd0]
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:328
  403a6a:	48 8d bb c8 00 00 00                            	lea    rdi,[rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403a71:	41 b8 70 00 00 00                               	mov    r8d,0x70
  403a77:	31 f6                                           	xor    esi,esi
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:327
  403a79:	c7 83 90 00 00 00 18 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x18
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403a83:	b9 01 00 00 00                                  	mov    ecx,0x1
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:383
  403a88:	5b                                              	pop    rbx
  403a89:	5d                                              	pop    rbp
  403a8a:	41 5c                                           	pop    r12
  403a8c:	41 5d                                           	pop    r13
  403a8e:	41 5e                                           	pop    r14
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403a90:	e9 8b e8 ff ff                                  	jmp    402320 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc@plt>
  403a95:	0f 1f 00                                        	nop    DWORD PTR [rax]
  403a98:	4c 8b 8b d0 00 00 00                            	mov    r9,QWORD PTR [rbx+0xd0]
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:323
  403a9f:	48 8d bb c8 00 00 00                            	lea    rdi,[rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403aa6:	b9 01 00 00 00                                  	mov    ecx,0x1
  403aab:	31 f6                                           	xor    esi,esi
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:322
  403aad:	c7 83 90 00 00 00 01 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x1
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403ab7:	44 0f be 42 01                                  	movsx  r8d,BYTE PTR [rdx+0x1]
  403abc:	4c 89 ca                                        	mov    rdx,r9
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:383
  403abf:	5b                                              	pop    rbx
  403ac0:	5d                                              	pop    rbp
  403ac1:	41 5c                                           	pop    r12
  403ac3:	41 5d                                           	pop    r13
  403ac5:	41 5e                                           	pop    r14
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403ac7:	e9 54 e8 ff ff                                  	jmp    402320 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc@plt>
  403acc:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:325
  403ad0:	40 80 fd 62                                     	cmp    bpl,0x62
  403ad4:	74 8d                                           	je     403a63 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0xa3>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:330
  403ad6:	40 80 fd 42                                     	cmp    bpl,0x42
  403ada:	0f 84 11 01 00 00                               	je     403bf1 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x231>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:337
  403ae0:	8d 45 bc                                        	lea    eax,[rbp-0x44]
  403ae3:	3c 33                                           	cmp    al,0x33
  403ae5:	0f 86 b5 00 00 00                               	jbe    403ba0 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x1e0>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:328
  403aeb:	4c 8d a3 c8 00 00 00                            	lea    r12,[rbx+0xc8]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:355
  403af2:	40 80 fd 78                                     	cmp    bpl,0x78
  403af6:	0f 84 8c 01 00 00                               	je     403c88 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x2c8>
  403afc:	40 80 fd 75                                     	cmp    bpl,0x75
  403b00:	0f 84 82 01 00 00                               	je     403c88 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x2c8>
_ZNKSt5ctypeIcE2isEtc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/x86_64-linux-gnu/bits/ctype_inline.h:44
  403b06:	48 8b 83 c0 00 00 00                            	mov    rax,QWORD PTR [rbx+0xc0]
  403b0d:	40 0f b6 ed                                     	movzx  ebp,bpl
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:902
  403b11:	48 8b 93 d0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd0]
_ZNKSt5ctypeIcE2isEtc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/x86_64-linux-gnu/bits/ctype_inline.h:44
  403b18:	48 8b 40 30                                     	mov    rax,QWORD PTR [rax+0x30]
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:370
  403b1c:	f6 44 68 01 08                                  	test   BYTE PTR [rax+rbp*2+0x1],0x8
  403b21:	0f 84 45 01 00 00                               	je     403c6c <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x2ac>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403b27:	45 89 f0                                        	mov    r8d,r14d
  403b2a:	b9 01 00 00 00                                  	mov    ecx,0x1
  403b2f:	31 f6                                           	xor    esi,esi
  403b31:	4c 89 e7                                        	mov    rdi,r12
  403b34:	e8 e7 e7 ff ff                                  	call   402320 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc@plt>
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:373
  403b39:	48 8b 83 b0 00 00 00                            	mov    rax,QWORD PTR [rbx+0xb0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:374
  403b40:	48 39 83 b8 00 00 00                            	cmp    QWORD PTR [rbx+0xb8],rax
  403b47:	75 2d                                           	jne    403b76 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x1b6>
  403b49:	eb 40                                           	jmp    403b8b <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x1cb>
  403b4b:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:375
  403b50:	48 8d 50 01                                     	lea    rdx,[rax+0x1]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEpLEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1159
  403b54:	4c 89 e7                                        	mov    rdi,r12
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:375
  403b57:	48 89 93 b0 00 00 00                            	mov    QWORD PTR [rbx+0xb0],rdx
  403b5e:	0f be 30                                        	movsx  esi,BYTE PTR [rax]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEpLEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1159
  403b61:	e8 2a e6 ff ff                                  	call   402190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc@plt>
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:373
  403b66:	48 8b 83 b0 00 00 00                            	mov    rax,QWORD PTR [rbx+0xb0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:374
  403b6d:	48 3b 83 b8 00 00 00                            	cmp    rax,QWORD PTR [rbx+0xb8]
  403b74:	74 15                                           	je     403b8b <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x1cb>
_ZNKSt5ctypeIcE2isEtc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/x86_64-linux-gnu/bits/ctype_inline.h:44 (discriminator 1)
  403b76:	48 8b 93 c0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xc0]
  403b7d:	0f b6 08                                        	movzx  ecx,BYTE PTR [rax]
  403b80:	48 8b 52 30                                     	mov    rdx,QWORD PTR [rdx+0x30]
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:374 (discriminator 1)
  403b84:	f6 44 4a 01 08                                  	test   BYTE PTR [rdx+rcx*2+0x1],0x8
  403b89:	75 c5                                           	jne    403b50 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x190>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:376
  403b8b:	c7 83 90 00 00 00 04 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x4
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:383
  403b95:	5b                                              	pop    rbx
  403b96:	5d                                              	pop    rbp
  403b97:	41 5c                                           	pop    r12
  403b99:	41 5d                                           	pop    r13
  403b9b:	41 5e                                           	pop    r14
  403b9d:	c3                                              	ret
  403b9e:	66 90                                           	xchg   ax,ax
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:336
  403ba0:	48 ba 01 80 08 00 01 80 08 00                   	movabs rdx,0x8800100088001
  403baa:	48 0f a3 c2                                     	bt     rdx,rax
  403bae:	73 6b                                           	jae    403c1b <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x25b>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:343
  403bb0:	c7 83 90 00 00 00 0e 00 00 00                   	mov    DWORD PTR [rbx+0x90],0xe
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403bba:	48 8b 93 d0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd0]
  403bc1:	45 89 f0                                        	mov    r8d,r14d
  403bc4:	31 f6                                           	xor    esi,esi
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:344
  403bc6:	48 8d bb c8 00 00 00                            	lea    rdi,[rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403bcd:	b9 01 00 00 00                                  	mov    ecx,0x1
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:383
  403bd2:	5b                                              	pop    rbx
  403bd3:	5d                                              	pop    rbp
  403bd4:	41 5c                                           	pop    r12
  403bd6:	41 5d                                           	pop    r13
  403bd8:	41 5e                                           	pop    r14
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403bda:	e9 41 e7 ff ff                                  	jmp    402320 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc@plt>
  403bdf:	90                                              	nop
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:935
  403be0:	31 d2                                           	xor    edx,edx
  403be2:	44 89 f6                                        	mov    esi,r14d
  403be5:	4c 89 e7                                        	mov    rdi,r12
  403be8:	ff d0                                           	call   rax
  403bea:	89 c1                                           	mov    ecx,eax
  403bec:	e9 2f fe ff ff                                  	jmp    403a20 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x60>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403bf1:	48 8b 93 d0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd0]
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:333
  403bf8:	48 8d bb c8 00 00 00                            	lea    rdi,[rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403bff:	41 b8 6e 00 00 00                               	mov    r8d,0x6e
  403c05:	31 f6                                           	xor    esi,esi
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:332
  403c07:	c7 83 90 00 00 00 18 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x18
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403c11:	b9 01 00 00 00                                  	mov    ecx,0x1
  403c16:	e9 a4 fe ff ff                                  	jmp    403abf <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0xff>
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:346
  403c1b:	40 80 fd 63                                     	cmp    bpl,0x63
  403c1f:	0f 85 c6 fe ff ff                               	jne    403aeb <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x12b>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:348
  403c25:	48 8b 83 b0 00 00 00                            	mov    rax,QWORD PTR [rbx+0xb0]
  403c2c:	48 3b 83 b8 00 00 00                            	cmp    rax,QWORD PTR [rbx+0xb8]
  403c33:	0f 84 e4 00 00 00                               	je     403d1d <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x35d>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:353
  403c39:	48 8d 50 01                                     	lea    rdx,[rax+0x1]
  403c3d:	48 8d bb c8 00 00 00                            	lea    rdi,[rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403c44:	b9 01 00 00 00                                  	mov    ecx,0x1
  403c49:	31 f6                                           	xor    esi,esi
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:353
  403c4b:	48 89 93 b0 00 00 00                            	mov    QWORD PTR [rbx+0xb0],rdx
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403c52:	48 8b 93 d0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd0]
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:352
  403c59:	c7 83 90 00 00 00 01 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x1
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  403c63:	44 0f be 00                                     	movsx  r8d,BYTE PTR [rax]
  403c67:	e9 53 fe ff ff                                  	jmp    403abf <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0xff>
  403c6c:	45 89 f0                                        	mov    r8d,r14d
  403c6f:	b9 01 00 00 00                                  	mov    ecx,0x1
  403c74:	31 f6                                           	xor    esi,esi
  403c76:	4c 89 e7                                        	mov    rdi,r12
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:380
  403c79:	c7 83 90 00 00 00 01 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x1
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1452
  403c83:	e9 37 fe ff ff                                  	jmp    403abf <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0xff>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  403c88:	48 8b 83 c8 00 00 00                            	mov    rax,QWORD PTR [rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:183
  403c8f:	48 c7 83 d0 00 00 00 00 00 00 00                	mov    QWORD PTR [rbx+0xd0],0x0
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  403c9a:	c6 00 00                                        	mov    BYTE PTR [rax],0x0
  403c9d:	31 c0                                           	xor    eax,eax
  403c9f:	40 80 fd 78                                     	cmp    bpl,0x78
  403ca3:	0f 95 c0                                        	setne  al
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:358
  403ca6:	31 ed                                           	xor    ebp,ebp
  403ca8:	44 8d 6c 00 02                                  	lea    r13d,[rax+rax*1+0x2]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:360
  403cad:	48 8b 83 b0 00 00 00                            	mov    rax,QWORD PTR [rbx+0xb0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:361
  403cb4:	48 3b 83 b8 00 00 00                            	cmp    rax,QWORD PTR [rbx+0xb8]
  403cbb:	74 42                                           	je     403cff <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x33f>
_ZNKSt5ctypeIcE2isEtc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/x86_64-linux-gnu/bits/ctype_inline.h:44 (discriminator 2)
  403cbd:	48 8b 93 c0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xc0]
  403cc4:	0f b6 08                                        	movzx  ecx,BYTE PTR [rax]
  403cc7:	48 8b 52 30                                     	mov    rdx,QWORD PTR [rdx+0x30]
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:361 (discriminator 2)
  403ccb:	f6 44 4a 01 10                                  	test   BYTE PTR [rdx+rcx*2+0x1],0x10
  403cd0:	74 2d                                           	je     403cff <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x33f>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:365
  403cd2:	48 8d 50 01                                     	lea    rdx,[rax+0x1]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEpLEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1159
  403cd6:	4c 89 e7                                        	mov    rdi,r12
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:358
  403cd9:	83 c5 01                                        	add    ebp,0x1
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:365
  403cdc:	48 89 93 b0 00 00 00                            	mov    QWORD PTR [rbx+0xb0],rdx
  403ce3:	0f be 30                                        	movsx  esi,BYTE PTR [rax]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEpLEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1159
  403ce6:	e8 a5 e4 ff ff                                  	call   402190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc@plt>
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:358
  403ceb:	41 39 ed                                        	cmp    r13d,ebp
  403cee:	75 bd                                           	jne    403cad <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x2ed>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:367
  403cf0:	c7 83 90 00 00 00 03 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x3
  403cfa:	e9 96 fe ff ff                                  	jmp    403b95 <_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv+0x1d5>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:362 (discriminator 3)
  403cff:	be 98 40 41 00                                  	mov    esi,0x414098
  403d04:	bf 02 00 00 00                                  	mov    edi,0x2
  403d09:	e8 52 e6 ff ff                                  	call   402360 <_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:314
  403d0e:	be 38 40 41 00                                  	mov    esi,0x414038
  403d13:	bf 02 00 00 00                                  	mov    edi,0x2
  403d18:	e8 43 e6 ff ff                                  	call   402360 <_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:349
  403d1d:	be 60 40 41 00                                  	mov    esi,0x414060
  403d22:	bf 02 00 00 00                                  	mov    edi,0x2
  403d27:	e8 34 e6 ff ff                                  	call   402360 <_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc>
  403d2c:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

0000000000403d30 <_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv>:
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:151
  403d30:	55                                              	push   rbp
  403d31:	48 89 fd                                        	mov    rbp,rdi
  403d34:	53                                              	push   rbx
_ZN9__gnu_cxx27__exchange_and_add_dispatchEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:83
  403d35:	bb 00 00 00 00                                  	mov    ebx,0x0
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:151
  403d3a:	48 83 ec 08                                     	sub    rsp,0x8
_ZN9__gnu_cxx27__exchange_and_add_dispatchEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:83
  403d3e:	48 85 db                                        	test   rbx,rbx
  403d41:	75 4d                                           	jne    403d90 <_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv+0x60>
_ZN9__gnu_cxx25__exchange_and_add_singleEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:68
  403d43:	8b 47 08                                        	mov    eax,DWORD PTR [rdi+0x8]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:69
  403d46:	8d 50 ff                                        	lea    edx,[rax-0x1]
  403d49:	89 57 08                                        	mov    DWORD PTR [rdi+0x8],edx
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:155
  403d4c:	83 f8 01                                        	cmp    eax,0x1
  403d4f:	74 0f                                           	je     403d60 <_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv+0x30>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:177
  403d51:	48 83 c4 08                                     	add    rsp,0x8
  403d55:	5b                                              	pop    rbx
  403d56:	5d                                              	pop    rbp
  403d57:	c3                                              	ret
  403d58:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:158
  403d60:	48 8b 45 00                                     	mov    rax,QWORD PTR [rbp+0x0]
  403d64:	48 89 ef                                        	mov    rdi,rbp
  403d67:	ff 50 10                                        	call   QWORD PTR [rax+0x10]
_ZN9__gnu_cxx27__exchange_and_add_dispatchEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:83
  403d6a:	48 85 db                                        	test   rbx,rbx
  403d6d:	75 31                                           	jne    403da0 <_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv+0x70>
_ZN9__gnu_cxx25__exchange_and_add_singleEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:68
  403d6f:	8b 45 0c                                        	mov    eax,DWORD PTR [rbp+0xc]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:69
  403d72:	8d 50 ff                                        	lea    edx,[rax-0x1]
  403d75:	89 55 0c                                        	mov    DWORD PTR [rbp+0xc],edx
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:170
  403d78:	83 f8 01                                        	cmp    eax,0x1
  403d7b:	75 d4                                           	jne    403d51 <_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv+0x21>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:174
  403d7d:	48 8b 45 00                                     	mov    rax,QWORD PTR [rbp+0x0]
  403d81:	48 89 ef                                        	mov    rdi,rbp
  403d84:	48 8b 40 18                                     	mov    rax,QWORD PTR [rax+0x18]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:177
  403d88:	48 83 c4 08                                     	add    rsp,0x8
  403d8c:	5b                                              	pop    rbx
  403d8d:	5d                                              	pop    rbp
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:174
  403d8e:	ff e0                                           	jmp    rax
_ZN9__gnu_cxx18__exchange_and_addEPVii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:50
  403d90:	b8 ff ff ff ff                                  	mov    eax,0xffffffff
  403d95:	f0 0f c1 47 08                                  	lock xadd DWORD PTR [rdi+0x8],eax
_ZN9__gnu_cxx27__exchange_and_add_dispatchEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:84
  403d9a:	eb b0                                           	jmp    403d4c <_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv+0x1c>
  403d9c:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]
_ZN9__gnu_cxx18__exchange_and_addEPVii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:50
  403da0:	b8 ff ff ff ff                                  	mov    eax,0xffffffff
  403da5:	f0 0f c1 45 0c                                  	lock xadd DWORD PTR [rbp+0xc],eax
_ZN9__gnu_cxx27__exchange_and_add_dispatchEPii():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/atomicity.h:84
  403daa:	eb cc                                           	jmp    403d78 <_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv+0x48>
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv():
  403dac:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

0000000000403db0 <_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv>:
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:140
  403db0:	41 54                                           	push   r12
  403db2:	49 89 fc                                        	mov    r12,rdi
  403db5:	53                                              	push   rbx
  403db6:	48 89 f3                                        	mov    rbx,rsi
  403db9:	48 83 ec 08                                     	sub    rsp,0x8
_ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_EC4ERKS8_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:169
  403dbd:	48 8b be 60 01 00 00                            	mov    rdi,QWORD PTR [rsi+0x160]
_ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_EmmEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:210
  403dc4:	48 3b be 68 01 00 00                            	cmp    rdi,QWORD PTR [rsi+0x168]
  403dcb:	74 33                                           	je     403e00 <_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv+0x50>
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:142
  403dcd:	48 8b 47 f8                                     	mov    rax,QWORD PTR [rdi-0x8]
  403dd1:	f3 0f 6f 4f e8                                  	movdqu xmm1,XMMWORD PTR [rdi-0x18]
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE8pop_backEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:1558
  403dd6:	48 83 ef 18                                     	sub    rdi,0x18
  403dda:	48 89 be 60 01 00 00                            	mov    QWORD PTR [rsi+0x160],rdi
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:142
  403de1:	49 89 44 24 10                                  	mov    QWORD PTR [r12+0x10],rax
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:145
  403de6:	4c 89 e0                                        	mov    rax,r12
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:142
  403de9:	41 0f 11 0c 24                                  	movups XMMWORD PTR [r12],xmm1
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:145
  403dee:	48 83 c4 08                                     	add    rsp,0x8
  403df2:	5b                                              	pop    rbx
  403df3:	41 5c                                           	pop    r12
  403df5:	c3                                              	ret
  403df6:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
_ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_E11_M_set_nodeEPS7_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:263
  403e00:	48 8b 86 78 01 00 00                            	mov    rax,QWORD PTR [rsi+0x178]
_ZN9__gnu_cxx13new_allocatorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEE10deallocateEPS6_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  403e07:	be f8 01 00 00                                  	mov    esi,0x1f8
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:142
  403e0c:	48 8b 40 f8                                     	mov    rax,QWORD PTR [rax-0x8]
  403e10:	f3 0f 6f 90 e0 01 00 00                         	movdqu xmm2,XMMWORD PTR [rax+0x1e0]
  403e18:	48 8b 80 f0 01 00 00                            	mov    rax,QWORD PTR [rax+0x1f0]
  403e1f:	41 0f 11 14 24                                  	movups XMMWORD PTR [r12],xmm2
  403e24:	49 89 44 24 10                                  	mov    QWORD PTR [r12+0x10],rax
_ZN9__gnu_cxx13new_allocatorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEE10deallocateEPS6_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  403e29:	e8 92 e3 ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE15_M_pop_back_auxEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/deque.tcc:561
  403e2e:	48 8b 93 78 01 00 00                            	mov    rdx,QWORD PTR [rbx+0x178]
_ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_E11_M_set_nodeEPS7_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:263
  403e35:	48 8b 42 f8                                     	mov    rax,QWORD PTR [rdx-0x8]
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE15_M_pop_back_auxEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/deque.tcc:561
  403e39:	48 83 ea 08                                     	sub    rdx,0x8
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/deque.tcc:562
  403e3d:	66 48 0f 6e e2                                  	movq   xmm4,rdx
  403e42:	48 8d 88 e0 01 00 00                            	lea    rcx,[rax+0x1e0]
  403e49:	66 48 0f 6e d8                                  	movq   xmm3,rax
_ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_E11_M_set_nodeEPS7_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:264
  403e4e:	48 05 f8 01 00 00                               	add    rax,0x1f8
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE15_M_pop_back_auxEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/deque.tcc:562
  403e54:	66 48 0f 6e c1                                  	movq   xmm0,rcx
  403e59:	66 0f 6c c3                                     	punpcklqdq xmm0,xmm3
  403e5d:	0f 11 83 60 01 00 00                            	movups XMMWORD PTR [rbx+0x160],xmm0
_ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_E11_M_set_nodeEPS7_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_deque.h:264
  403e64:	66 48 0f 6e c0                                  	movq   xmm0,rax
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:145
  403e69:	4c 89 e0                                        	mov    rax,r12
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE15_M_pop_back_auxEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/deque.tcc:562
  403e6c:	66 0f 6c c4                                     	punpcklqdq xmm0,xmm4
  403e70:	0f 11 83 70 01 00 00                            	movups XMMWORD PTR [rbx+0x170],xmm0
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_compiler.h:145
  403e77:	48 83 c4 08                                     	add    rsp,0x8
  403e7b:	5b                                              	pop    rbx
  403e7c:	41 5c                                           	pop    r12
  403e7e:	c3                                              	ret
  403e7f:	90                                              	nop

0000000000403e80 <_ZNSt8__detail6_StateIcED1Ev>:
_ZNSt8__detail6_StateIcED2Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_automaton.h:158
  403e80:	83 3f 0b                                        	cmp    DWORD PTR [rdi],0xb
  403e83:	74 0b                                           	je     403e90 <_ZNSt8__detail6_StateIcED1Ev+0x10>
  403e85:	c3                                              	ret
  403e86:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
_ZNSt14_Function_baseD4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:244
  403e90:	48 8b 47 20                                     	mov    rax,QWORD PTR [rdi+0x20]
  403e94:	48 85 c0                                        	test   rax,rax
  403e97:	74 ec                                           	je     403e85 <_ZNSt8__detail6_StateIcED1Ev+0x5>
_ZNSt8__detail6_StateIcED2Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_automaton.h:156
  403e99:	48 83 ec 08                                     	sub    rsp,0x8
_ZNSt14_Function_baseD4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:245
  403e9d:	48 83 c7 10                                     	add    rdi,0x10
  403ea1:	ba 03 00 00 00                                  	mov    edx,0x3
  403ea6:	48 89 fe                                        	mov    rsi,rdi
  403ea9:	ff d0                                           	call   rax
_ZNSt8__detail6_StateIcED2Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_automaton.h:160
  403eab:	48 83 c4 08                                     	add    rsp,0x8
  403eaf:	c3                                              	ret

0000000000403eb0 <_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv>:
_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:558
  403eb0:	41 54                                           	push   r12
  403eb2:	49 89 fc                                        	mov    r12,rdi
_ZNSt7__cxx1112regex_traitsIcED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:80
  403eb5:	48 83 c7 60                                     	add    rdi,0x60
_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:558
  403eb9:	55                                              	push   rbp
  403eba:	53                                              	push   rbx
_ZNSt7__cxx1112regex_traitsIcED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex.h:80
  403ebb:	e8 c0 e3 ff ff                                  	call   402280 <_ZNSt6localeD1Ev@plt>
_ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:680
  403ec0:	49 8b 5c 24 50                                  	mov    rbx,QWORD PTR [r12+0x50]
  403ec5:	49 8b 6c 24 48                                  	mov    rbp,QWORD PTR [r12+0x48]
_ZNSt12_Destroy_auxILb0EE9__destroyIPNSt8__detail6_StateIcEEEEvT_S6_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_construct.h:151
  403eca:	48 39 eb                                        	cmp    rbx,rbp
  403ecd:	74 17                                           	je     403ee6 <_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv+0x36>
  403ecf:	90                                              	nop
_ZSt8_DestroyINSt8__detail6_StateIcEEEvPT_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_construct.h:140
  403ed0:	48 89 ef                                        	mov    rdi,rbp
_ZNSt12_Destroy_auxILb0EE9__destroyIPNSt8__detail6_StateIcEEEEvT_S6_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_construct.h:151
  403ed3:	48 83 c5 30                                     	add    rbp,0x30
_ZSt8_DestroyINSt8__detail6_StateIcEEEvPT_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_construct.h:140
  403ed7:	e8 a4 ff ff ff                                  	call   403e80 <_ZNSt8__detail6_StateIcED1Ev>
_ZNSt12_Destroy_auxILb0EE9__destroyIPNSt8__detail6_StateIcEEEEvT_S6_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_construct.h:151
  403edc:	48 39 eb                                        	cmp    rbx,rbp
  403edf:	75 ef                                           	jne    403ed0 <_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv+0x20>
_ZNSt12_Vector_baseINSt8__detail6_StateIcEESaIS2_EED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:336
  403ee1:	49 8b 6c 24 48                                  	mov    rbp,QWORD PTR [r12+0x48]
_ZNSt12_Vector_baseINSt8__detail6_StateIcEESaIS2_EE13_M_deallocateEPS2_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:353
  403ee6:	48 85 ed                                        	test   rbp,rbp
  403ee9:	74 10                                           	je     403efb <_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv+0x4b>
_ZNSt12_Vector_baseINSt8__detail6_StateIcEESaIS2_EED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:336
  403eeb:	49 8b 74 24 58                                  	mov    rsi,QWORD PTR [r12+0x58]
_ZN9__gnu_cxx13new_allocatorINSt8__detail6_StateIcEEE10deallocateEPS3_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  403ef0:	48 89 ef                                        	mov    rdi,rbp
_ZNSt12_Vector_baseINSt8__detail6_StateIcEESaIS2_EED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:336
  403ef3:	48 29 ee                                        	sub    rsi,rbp
_ZN9__gnu_cxx13new_allocatorINSt8__detail6_StateIcEEE10deallocateEPS3_m():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  403ef6:	e8 c5 e2 ff ff                                  	call   4021c0 <_ZdlPvm@plt>
_ZNSt6vectorImSaImEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:680
  403efb:	49 8b 7c 24 10                                  	mov    rdi,QWORD PTR [r12+0x10]
_ZNSt12_Vector_baseImSaImEE13_M_deallocateEPmm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:353
  403f00:	48 85 ff                                        	test   rdi,rdi
  403f03:	74 1b                                           	je     403f20 <_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv+0x70>
_ZNSt12_Vector_baseImSaImEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:336
  403f05:	49 8b 74 24 20                                  	mov    rsi,QWORD PTR [r12+0x20]
_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:561
  403f0a:	5b                                              	pop    rbx
  403f0b:	5d                                              	pop    rbp
  403f0c:	41 5c                                           	pop    r12
_ZNSt12_Vector_baseImSaImEED4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:336
  403f0e:	48 29 fe                                        	sub    rsi,rdi
_ZN9__gnu_cxx13new_allocatorImE10deallocateEPmm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:133
  403f11:	e9 aa e2 ff ff                                  	jmp    4021c0 <_ZdlPvm@plt>
  403f16:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
_ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/shared_ptr_base.h:561
  403f20:	5b                                              	pop    rbx
  403f21:	5d                                              	pop    rbp
  403f22:	41 5c                                           	pop    r12
  403f24:	c3                                              	ret
  403f25:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  403f2f:	90                                              	nop

0000000000403f30 <_ZNSt8__detail6_StateIcEC1EOS1_>:
_ZNSt8__detail6_StateIcEC2EOS1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_automaton.h:146
  403f30:	f3 0f 6f 0e                                     	movdqu xmm1,XMMWORD PTR [rsi]
  403f34:	f3 0f 6f 46 10                                  	movdqu xmm0,XMMWORD PTR [rsi+0x10]
  403f39:	f3 0f 6f 56 20                                  	movdqu xmm2,XMMWORD PTR [rsi+0x20]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_automaton.h:148
  403f3e:	83 3e 0b                                        	cmp    DWORD PTR [rsi],0xb
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_automaton.h:146
  403f41:	0f 11 0f                                        	movups XMMWORD PTR [rdi],xmm1
  403f44:	0f 11 47 10                                     	movups XMMWORD PTR [rdi+0x10],xmm0
  403f48:	0f 11 57 20                                     	movups XMMWORD PTR [rdi+0x20],xmm2
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_automaton.h:148
  403f4c:	74 02                                           	je     403f50 <_ZNSt8__detail6_StateIcEC1EOS1_+0x20>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_automaton.h:151
  403f4e:	c3                                              	ret
  403f4f:	90                                              	nop
_ZNSt14_Function_baseC4Ev():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/std_function.h:240
  403f50:	48 c7 47 20 00 00 00 00                         	mov    QWORD PTR [rdi+0x20],0x0
_ZSt4swapIPFbRSt9_Any_dataRKS0_St18_Manager_operationEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISB_ESt18is_move_assignableISB_EEE5valueEvE4typeERSB_SL_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:197
  403f58:	48 8b 46 20                                     	mov    rax,QWORD PTR [rsi+0x20]
_ZSt4swapISt9_Any_dataENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS5_ESt18is_move_assignableIS5_EEE5valueEvE4typeERS5_SF_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:197
  403f5c:	f3 0f 6f 5e 10                                  	movdqu xmm3,XMMWORD PTR [rsi+0x10]
_ZSt4swapIPFbRKSt9_Any_dataOcEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISA_ESt18is_move_assignableISA_EEE5valueEvE4typeERSA_SK_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:198
  403f61:	48 8b 57 28                                     	mov    rdx,QWORD PTR [rdi+0x28]
_ZSt4swapISt9_Any_dataENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS5_ESt18is_move_assignableIS5_EEE5valueEvE4typeERS5_SF_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:198
  403f65:	0f 11 46 10                                     	movups XMMWORD PTR [rsi+0x10],xmm0
_ZSt4swapIPFbRSt9_Any_dataRKS0_St18_Manager_operationEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISB_ESt18is_move_assignableISB_EEE5valueEvE4typeERSB_SL_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:199
  403f69:	48 89 47 20                                     	mov    QWORD PTR [rdi+0x20],rax
_ZSt4swapIPFbRKSt9_Any_dataOcEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISA_ESt18is_move_assignableISA_EEE5valueEvE4typeERSA_SK_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:197
  403f6d:	48 8b 46 28                                     	mov    rax,QWORD PTR [rsi+0x28]
_ZSt4swapIPFbRSt9_Any_dataRKS0_St18_Manager_operationEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISB_ESt18is_move_assignableISB_EEE5valueEvE4typeERSB_SL_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:198
  403f71:	48 c7 46 20 00 00 00 00                         	mov    QWORD PTR [rsi+0x20],0x0
_ZSt4swapIPFbRKSt9_Any_dataOcEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISA_ESt18is_move_assignableISA_EEE5valueEvE4typeERSA_SK_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:198
  403f79:	48 89 56 28                                     	mov    QWORD PTR [rsi+0x28],rdx
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:199
  403f7d:	48 89 47 28                                     	mov    QWORD PTR [rdi+0x28],rax
_ZSt4swapISt9_Any_dataENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS5_ESt18is_move_assignableIS5_EEE5valueEvE4typeERS5_SF_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/move.h:197
  403f81:	0f 11 5f 10                                     	movups XMMWORD PTR [rdi+0x10],xmm3
_ZNSt8__detail6_StateIcEC2EOS1_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_automaton.h:151
  403f85:	c3                                              	ret
  403f86:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]

0000000000403f90 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv>:
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:431
  403f90:	41 57                                           	push   r15
  403f92:	41 56                                           	push   r14
  403f94:	41 55                                           	push   r13
  403f96:	41 54                                           	push   r12
  403f98:	55                                              	push   rbp
  403f99:	53                                              	push   rbx
  403f9a:	48 89 fb                                        	mov    rbx,rdi
  403f9d:	48 83 ec 08                                     	sub    rsp,0x8
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:434
  403fa1:	48 8b 87 b0 00 00 00                            	mov    rax,QWORD PTR [rdi+0xb0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:435
  403fa8:	48 8b af c0 00 00 00                            	mov    rbp,QWORD PTR [rdi+0xc0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:434
  403faf:	48 8d 50 01                                     	lea    rdx,[rax+0x1]
  403fb3:	48 89 97 b0 00 00 00                            	mov    QWORD PTR [rdi+0xb0],rdx
  403fba:	44 0f be 30                                     	movsx  r14d,BYTE PTR [rax]
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:933
  403fbe:	45 0f b6 e6                                     	movzx  r12d,r14b
  403fc2:	45 89 f5                                        	mov    r13d,r14d
  403fc5:	42 0f b6 84 25 39 01 00 00                      	movzx  eax,BYTE PTR [rbp+r12*1+0x139]
  403fce:	89 c1                                           	mov    ecx,eax
  403fd0:	84 c0                                           	test   al,al
  403fd2:	75 23                                           	jne    403ff7 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x67>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:935
  403fd4:	48 8b 45 00                                     	mov    rax,QWORD PTR [rbp+0x0]
  403fd8:	44 89 f1                                        	mov    ecx,r14d
  403fdb:	48 8b 40 40                                     	mov    rax,QWORD PTR [rax+0x40]
  403fdf:	48 3d 90 30 40 00                               	cmp    rax,0x403090
  403fe5:	0f 85 2d 02 00 00                               	jne    404218 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x288>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:936
  403feb:	84 c9                                           	test   cl,cl
  403fed:	74 08                                           	je     403ff7 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x67>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:937
  403fef:	42 88 8c 25 39 01 00 00                         	mov    BYTE PTR [rbp+r12*1+0x139],cl
_ZNSt8__detail12_ScannerBase14_M_find_escapeEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.h:117
  403ff7:	48 8b 83 98 00 00 00                            	mov    rax,QWORD PTR [rbx+0x98]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.h:118
  403ffe:	0f b6 10                                        	movzx  edx,BYTE PTR [rax]
  404001:	84 d2                                           	test   dl,dl
  404003:	75 17                                           	jne    40401c <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x8c>
  404005:	eb 59                                           	jmp    404060 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0xd0>
  404007:	66 0f 1f 84 00 00 00 00 00                      	nop    WORD PTR [rax+rax*1+0x0]
  404010:	0f b6 50 02                                     	movzx  edx,BYTE PTR [rax+0x2]
  404014:	48 83 c0 02                                     	add    rax,0x2
  404018:	84 d2                                           	test   dl,dl
  40401a:	74 44                                           	je     404060 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0xd0>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.h:119
  40401c:	38 ca                                           	cmp    dl,cl
  40401e:	75 f0                                           	jne    404010 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x80>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  404020:	48 8b 93 d0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd0]
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:440
  404027:	48 8d bb c8 00 00 00                            	lea    rdi,[rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  40402e:	b9 01 00 00 00                                  	mov    ecx,0x1
  404033:	31 f6                                           	xor    esi,esi
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:439
  404035:	c7 83 90 00 00 00 01 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x1
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  40403f:	44 0f be 40 01                                  	movsx  r8d,BYTE PTR [rax+0x1]
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:462
  404044:	48 83 c4 08                                     	add    rsp,0x8
  404048:	5b                                              	pop    rbx
  404049:	5d                                              	pop    rbp
  40404a:	41 5c                                           	pop    r12
  40404c:	41 5d                                           	pop    r13
  40404e:	41 5e                                           	pop    r14
  404050:	41 5f                                           	pop    r15
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  404052:	e9 c9 e2 ff ff                                  	jmp    402320 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc@plt>
  404057:	66 0f 1f 84 00 00 00 00 00                      	nop    WORD PTR [rax+rax*1+0x0]
_ZNKSt5ctypeIcE2isEtc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/x86_64-linux-gnu/bits/ctype_inline.h:44
  404060:	48 8b 93 c0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xc0]
  404067:	41 0f b6 c5                                     	movzx  eax,r13b
  40406b:	48 8b 52 30                                     	mov    rdx,QWORD PTR [rdx+0x30]
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:445
  40406f:	f6 44 42 01 08                                  	test   BYTE PTR [rdx+rax*2+0x1],0x8
  404074:	0f 84 c3 01 00 00                               	je     40423d <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x2ad>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:445 (discriminator 1)
  40407a:	41 83 ed 38                                     	sub    r13d,0x38
  40407e:	41 80 fd 01                                     	cmp    r13b,0x1
  404082:	0f 86 b5 01 00 00                               	jbe    40423d <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x2ad>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  404088:	48 8b 93 d0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd0]
  40408f:	45 89 f0                                        	mov    r8d,r14d
  404092:	b9 01 00 00 00                                  	mov    ecx,0x1
  404097:	31 f6                                           	xor    esi,esi
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:447
  404099:	4c 8d a3 c8 00 00 00                            	lea    r12,[rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  4040a0:	4c 89 e7                                        	mov    rdi,r12
  4040a3:	e8 78 e2 ff ff                                  	call   402320 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc@plt>
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:450
  4040a8:	48 8b 83 b0 00 00 00                            	mov    rax,QWORD PTR [rbx+0xb0]
  4040af:	48 39 83 b8 00 00 00                            	cmp    QWORD PTR [rbx+0xb8],rax
  4040b6:	0f 84 fd 00 00 00                               	je     4041b9 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x229>
_ZNKSt5ctypeIcE2isEtc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/x86_64-linux-gnu/bits/ctype_inline.h:44
  4040bc:	48 8b 8b c0 00 00 00                            	mov    rcx,QWORD PTR [rbx+0xc0]
  4040c3:	0f b6 30                                        	movzx  esi,BYTE PTR [rax]
  4040c6:	48 8b 49 30                                     	mov    rcx,QWORD PTR [rcx+0x30]
  4040ca:	48 89 f2                                        	mov    rdx,rsi
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:453
  4040cd:	f6 44 71 01 08                                  	test   BYTE PTR [rcx+rsi*2+0x1],0x8
  4040d2:	0f 84 e1 00 00 00                               	je     4041b9 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x229>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:452
  4040d8:	83 ea 38                                        	sub    edx,0x38
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:453
  4040db:	80 fa 01                                        	cmp    dl,0x1
  4040de:	0f 86 d5 00 00 00                               	jbe    4041b9 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x229>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:455
  4040e4:	48 8d 50 01                                     	lea    rdx,[rax+0x1]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:902
  4040e8:	48 8b ab d0 00 00 00                            	mov    rbp,QWORD PTR [rbx+0xd0]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:203
  4040ef:	4c 8d ab d8 00 00 00                            	lea    r13,[rbx+0xd8]
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:455
  4040f6:	48 89 93 b0 00 00 00                            	mov    QWORD PTR [rbx+0xb0],rdx
  4040fd:	44 0f b6 38                                     	movzx  r15d,BYTE PTR [rax]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  404101:	48 8b 83 c8 00 00 00                            	mov    rax,QWORD PTR [rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1323
  404108:	4c 8d 75 01                                     	lea    r14,[rbp+0x1]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8capacityEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:966
  40410c:	4c 39 e8                                        	cmp    rax,r13
  40410f:	0f 84 14 01 00 00                               	je     404229 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x299>
  404115:	48 8b 93 d8 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1323
  40411c:	49 39 d6                                        	cmp    r14,rdx
  40411f:	0f 87 ad 00 00 00                               	ja     4041d2 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x242>
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  404125:	44 88 3c 28                                     	mov    BYTE PTR [rax+rbp*1],r15b
  404129:	48 8b 83 c8 00 00 00                            	mov    rax,QWORD PTR [rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:183
  404130:	4c 89 b3 d0 00 00 00                            	mov    QWORD PTR [rbx+0xd0],r14
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  404137:	c6 44 28 01 00                                  	mov    BYTE PTR [rax+rbp*1+0x1],0x0
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:450
  40413c:	48 8b 83 b0 00 00 00                            	mov    rax,QWORD PTR [rbx+0xb0]
  404143:	48 3b 83 b8 00 00 00                            	cmp    rax,QWORD PTR [rbx+0xb8]
  40414a:	74 6d                                           	je     4041b9 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x229>
_ZNKSt5ctypeIcE2isEtc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/x86_64-linux-gnu/bits/ctype_inline.h:44
  40414c:	48 8b 8b c0 00 00 00                            	mov    rcx,QWORD PTR [rbx+0xc0]
  404153:	0f b6 30                                        	movzx  esi,BYTE PTR [rax]
  404156:	48 8b 49 30                                     	mov    rcx,QWORD PTR [rcx+0x30]
  40415a:	48 89 f2                                        	mov    rdx,rsi
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:453
  40415d:	f6 44 71 01 08                                  	test   BYTE PTR [rcx+rsi*2+0x1],0x8
  404162:	74 55                                           	je     4041b9 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x229>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:452
  404164:	83 ea 38                                        	sub    edx,0x38
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:453
  404167:	80 fa 01                                        	cmp    dl,0x1
  40416a:	76 4d                                           	jbe    4041b9 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x229>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:455
  40416c:	48 8d 50 01                                     	lea    rdx,[rax+0x1]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:902
  404170:	48 8b ab d0 00 00 00                            	mov    rbp,QWORD PTR [rbx+0xd0]
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:455
  404177:	48 89 93 b0 00 00 00                            	mov    QWORD PTR [rbx+0xb0],rdx
  40417e:	44 0f b6 38                                     	movzx  r15d,BYTE PTR [rax]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  404182:	48 8b 83 c8 00 00 00                            	mov    rax,QWORD PTR [rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1323
  404189:	4c 8d 75 01                                     	lea    r14,[rbp+0x1]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8capacityEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:966
  40418d:	49 39 c5                                        	cmp    r13,rax
  404190:	0f 84 9d 00 00 00                               	je     404233 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x2a3>
  404196:	48 8b 93 d8 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1323
  40419d:	49 39 d6                                        	cmp    r14,rdx
  4041a0:	77 51                                           	ja     4041f3 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x263>
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  4041a2:	44 88 3c 28                                     	mov    BYTE PTR [rax+rbp*1],r15b
  4041a6:	48 8b 83 c8 00 00 00                            	mov    rax,QWORD PTR [rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:183
  4041ad:	4c 89 b3 d0 00 00 00                            	mov    QWORD PTR [rbx+0xd0],r14
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  4041b4:	c6 44 28 01 00                                  	mov    BYTE PTR [rax+rbp*1+0x1],0x0
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:456
  4041b9:	c7 83 90 00 00 00 02 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x2
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:462
  4041c3:	48 83 c4 08                                     	add    rsp,0x8
  4041c7:	5b                                              	pop    rbx
  4041c8:	5d                                              	pop    rbp
  4041c9:	41 5c                                           	pop    r12
  4041cb:	41 5d                                           	pop    r13
  4041cd:	41 5e                                           	pop    r14
  4041cf:	41 5f                                           	pop    r15
  4041d1:	c3                                              	ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1324
  4041d2:	41 b8 01 00 00 00                               	mov    r8d,0x1
  4041d8:	31 c9                                           	xor    ecx,ecx
  4041da:	31 d2                                           	xor    edx,edx
  4041dc:	48 89 ee                                        	mov    rsi,rbp
  4041df:	4c 89 e7                                        	mov    rdi,r12
  4041e2:	e8 19 e1 ff ff                                  	call   402300 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm@plt>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  4041e7:	48 8b 83 c8 00 00 00                            	mov    rax,QWORD PTR [rbx+0xc8]
  4041ee:	e9 32 ff ff ff                                  	jmp    404125 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x195>
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1324
  4041f3:	41 b8 01 00 00 00                               	mov    r8d,0x1
  4041f9:	31 c9                                           	xor    ecx,ecx
  4041fb:	31 d2                                           	xor    edx,edx
  4041fd:	48 89 ee                                        	mov    rsi,rbp
  404200:	4c 89 e7                                        	mov    rdi,r12
  404203:	e8 f8 e0 ff ff                                  	call   402300 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm@plt>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  404208:	48 8b 83 c8 00 00 00                            	mov    rax,QWORD PTR [rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  40420f:	eb 91                                           	jmp    4041a2 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x212>
  404211:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:935
  404218:	31 d2                                           	xor    edx,edx
  40421a:	44 89 f6                                        	mov    esi,r14d
  40421d:	48 89 ef                                        	mov    rdi,rbp
  404220:	ff d0                                           	call   rax
  404222:	89 c1                                           	mov    ecx,eax
  404224:	e9 c2 fd ff ff                                  	jmp    403feb <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x5b>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8capacityEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:966
  404229:	ba 0f 00 00 00                                  	mov    edx,0xf
  40422e:	e9 e9 fe ff ff                                  	jmp    40411c <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x18c>
  404233:	ba 0f 00 00 00                                  	mov    edx,0xf
  404238:	e9 60 ff ff ff                                  	jmp    40419d <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv+0x20d>
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:460
  40423d:	be c6 40 41 00                                  	mov    esi,0x4140c6
  404242:	bf 02 00 00 00                                  	mov    edi,0x2
  404247:	e8 14 e1 ff ff                                  	call   402360 <_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc>
  40424c:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]

0000000000404250 <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv>:
_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:389
  404250:	41 56                                           	push   r14
  404252:	41 55                                           	push   r13
  404254:	41 54                                           	push   r12
  404256:	55                                              	push   rbp
  404257:	53                                              	push   rbx
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:392
  404258:	48 8b 87 b0 00 00 00                            	mov    rax,QWORD PTR [rdi+0xb0]
  40425f:	48 3b 87 b8 00 00 00                            	cmp    rax,QWORD PTR [rdi+0xb8]
  404266:	0f 84 43 01 00 00                               	je     4043af <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv+0x15f>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:396
  40426c:	44 0f be 28                                     	movsx  r13d,BYTE PTR [rax]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:397
  404270:	4c 8b a7 c0 00 00 00                            	mov    r12,QWORD PTR [rdi+0xc0]
  404277:	48 89 fd                                        	mov    rbp,rdi
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:933
  40427a:	45 0f b6 f5                                     	movzx  r14d,r13b
_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:396
  40427e:	44 89 eb                                        	mov    ebx,r13d
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:933
  404281:	43 0f be b4 34 39 01 00 00                      	movsx  esi,BYTE PTR [r12+r14*1+0x139]
  40428a:	40 84 f6                                        	test   sil,sil
  40428d:	75 27                                           	jne    4042b6 <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv+0x66>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:935
  40428f:	49 8b 04 24                                     	mov    rax,QWORD PTR [r12]
_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:397
  404293:	44 89 ee                                        	mov    esi,r13d
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:935
  404296:	48 8b 48 40                                     	mov    rcx,QWORD PTR [rax+0x40]
  40429a:	44 89 e8                                        	mov    eax,r13d
  40429d:	48 81 f9 90 30 40 00                            	cmp    rcx,0x403090
  4042a4:	0f 85 f6 00 00 00                               	jne    4043a0 <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv+0x150>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:936
  4042aa:	84 c0                                           	test   al,al
  4042ac:	74 08                                           	je     4042b6 <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv+0x66>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:937
  4042ae:	43 88 84 34 39 01 00 00                         	mov    BYTE PTR [r12+r14*1+0x139],al
strchr():
/usr/include/string.h:221
  4042b6:	48 8b bd a0 00 00 00                            	mov    rdi,QWORD PTR [rbp+0xa0]
  4042bd:	e8 ce dd ff ff                                  	call   402090 <strchr@plt>
_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:399
  4042c2:	48 85 c0                                        	test   rax,rax
  4042c5:	74 05                                           	je     4042cc <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv+0x7c>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:399 (discriminator 1)
  4042c7:	80 38 00                                        	cmp    BYTE PTR [rax],0x0
  4042ca:	75 6c                                           	jne    404338 <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv+0xe8>
_ZNKSt8__detail12_ScannerBase9_M_is_awkEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.h:146
  4042cc:	8b 85 8c 00 00 00                               	mov    eax,DWORD PTR [rbp+0x8c]
_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:405
  4042d2:	a8 80                                           	test   al,0x80
  4042d4:	0f 85 ae 00 00 00                               	jne    404388 <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv+0x138>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:902
  4042da:	48 8b 95 d0 00 00 00                            	mov    rdx,QWORD PTR [rbp+0xd0]
_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:402
  4042e1:	48 8d bd c8 00 00 00                            	lea    rdi,[rbp+0xc8]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:410
  4042e8:	a9 20 01 00 00                                  	test   eax,0x120
  4042ed:	74 1a                                           	je     404309 <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv+0xb9>
_ZNKSt5ctypeIcE2isEtc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/x86_64-linux-gnu/bits/ctype_inline.h:44 (discriminator 1)
  4042ef:	48 8b 8d c0 00 00 00                            	mov    rcx,QWORD PTR [rbp+0xc0]
  4042f6:	0f b6 c3                                        	movzx  eax,bl
  4042f9:	48 8b 49 30                                     	mov    rcx,QWORD PTR [rcx+0x30]
_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:410 (discriminator 1)
  4042fd:	f6 44 41 01 08                                  	test   BYTE PTR [rcx+rax*2+0x1],0x8
  404302:	74 05                                           	je     404309 <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv+0xb9>
  404304:	80 fb 30                                        	cmp    bl,0x30
  404307:	75 5f                                           	jne    404368 <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv+0x118>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:422 (discriminator 6)
  404309:	c7 85 90 00 00 00 01 00 00 00                   	mov    DWORD PTR [rbp+0x90],0x1
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453 (discriminator 6)
  404313:	45 89 e8                                        	mov    r8d,r13d
  404316:	b9 01 00 00 00                                  	mov    ecx,0x1
  40431b:	31 f6                                           	xor    esi,esi
  40431d:	e8 fe df ff ff                                  	call   402320 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc@plt>
_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:426
  404322:	48 83 85 b0 00 00 00 01                         	add    QWORD PTR [rbp+0xb0],0x1
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:427
  40432a:	5b                                              	pop    rbx
  40432b:	5d                                              	pop    rbp
  40432c:	41 5c                                           	pop    r12
  40432e:	41 5d                                           	pop    r13
  404330:	41 5e                                           	pop    r14
  404332:	c3                                              	ret
  404333:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:401
  404338:	c7 85 90 00 00 00 01 00 00 00                   	mov    DWORD PTR [rbp+0x90],0x1
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  404342:	48 8b 95 d0 00 00 00                            	mov    rdx,QWORD PTR [rbp+0xd0]
  404349:	45 89 e8                                        	mov    r8d,r13d
  40434c:	31 f6                                           	xor    esi,esi
_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:402
  40434e:	48 8d bd c8 00 00 00                            	lea    rdi,[rbp+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  404355:	b9 01 00 00 00                                  	mov    ecx,0x1
  40435a:	e8 c1 df ff ff                                  	call   402320 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc@plt>
  40435f:	eb c1                                           	jmp    404322 <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv+0xd2>
  404361:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:412
  404368:	c7 85 90 00 00 00 04 00 00 00                   	mov    DWORD PTR [rbp+0x90],0x4
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  404372:	45 89 e8                                        	mov    r8d,r13d
  404375:	b9 01 00 00 00                                  	mov    ecx,0x1
  40437a:	31 f6                                           	xor    esi,esi
  40437c:	e8 9f df ff ff                                  	call   402320 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc@plt>
  404381:	eb 9f                                           	jmp    404322 <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv+0xd2>
  404383:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]
_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:427
  404388:	5b                                              	pop    rbx
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:407
  404389:	48 89 ef                                        	mov    rdi,rbp
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:427
  40438c:	5d                                              	pop    rbp
  40438d:	41 5c                                           	pop    r12
  40438f:	41 5d                                           	pop    r13
  404391:	41 5e                                           	pop    r14
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:407
  404393:	e9 f8 fb ff ff                                  	jmp    403f90 <_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv>
  404398:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:935
  4043a0:	31 d2                                           	xor    edx,edx
  4043a2:	4c 89 e7                                        	mov    rdi,r12
  4043a5:	ff d1                                           	call   rcx
_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:397
  4043a7:	0f be f0                                        	movsx  esi,al
  4043aa:	e9 fb fe ff ff                                  	jmp    4042aa <_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv+0x5a>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:393
  4043af:	be 38 40 41 00                                  	mov    esi,0x414038
  4043b4:	bf 02 00 00 00                                  	mov    edi,0x2
  4043b9:	e8 a2 df ff ff                                  	call   402360 <_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc>
  4043be:	66 90                                           	xchg   ax,ax

00000000004043c0 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv>:
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:96
  4043c0:	41 56                                           	push   r14
  4043c2:	41 55                                           	push   r13
  4043c4:	41 54                                           	push   r12
  4043c6:	55                                              	push   rbp
  4043c7:	53                                              	push   rbx
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:99
  4043c8:	48 8b 87 b0 00 00 00                            	mov    rax,QWORD PTR [rdi+0xb0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:96
  4043cf:	48 89 fb                                        	mov    rbx,rdi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:101
  4043d2:	4c 8b a7 c0 00 00 00                            	mov    r12,QWORD PTR [rdi+0xc0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:99
  4043d9:	48 8d 50 01                                     	lea    rdx,[rax+0x1]
  4043dd:	48 89 97 b0 00 00 00                            	mov    QWORD PTR [rdi+0xb0],rdx
  4043e4:	44 0f be 28                                     	movsx  r13d,BYTE PTR [rax]
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:933
  4043e8:	45 0f b6 f5                                     	movzx  r14d,r13b
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:99
  4043ec:	44 89 ed                                        	mov    ebp,r13d
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:933
  4043ef:	43 0f be b4 34 39 01 00 00                      	movsx  esi,BYTE PTR [r12+r14*1+0x139]
  4043f8:	40 84 f6                                        	test   sil,sil
  4043fb:	75 27                                           	jne    404424 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x64>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:935
  4043fd:	49 8b 04 24                                     	mov    rax,QWORD PTR [r12]
  404401:	44 89 ee                                        	mov    esi,r13d
  404404:	48 8b 48 40                                     	mov    rcx,QWORD PTR [rax+0x40]
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:99
  404408:	44 89 e8                                        	mov    eax,r13d
  40440b:	48 81 f9 90 30 40 00                            	cmp    rcx,0x403090
  404412:	0f 85 30 02 00 00                               	jne    404648 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x288>
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:936
  404418:	3c 20                                           	cmp    al,0x20
  40441a:	74 08                                           	je     404424 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x64>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:937
  40441c:	43 88 84 34 39 01 00 00                         	mov    BYTE PTR [r12+r14*1+0x139],al
strchr():
/usr/include/string.h:221
  404424:	48 8b bb a0 00 00 00                            	mov    rdi,QWORD PTR [rbx+0xa0]
  40442b:	e8 60 dc ff ff                                  	call   402090 <strchr@plt>
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:101
  404430:	48 85 c0                                        	test   rax,rax
  404433:	0f 84 5f 02 00 00                               	je     404698 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x2d8>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:107
  404439:	40 80 fd 5c                                     	cmp    bpl,0x5c
  40443d:	0f 84 d5 00 00 00                               	je     404518 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x158>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:124
  404443:	40 80 fd 28                                     	cmp    bpl,0x28
  404447:	0f 84 a3 00 00 00                               	je     4044f0 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x130>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:160
  40444d:	40 80 fd 29                                     	cmp    bpl,0x29
  404451:	0f 84 d9 01 00 00                               	je     404630 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x270>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:162
  404457:	40 80 fd 5b                                     	cmp    bpl,0x5b
  40445b:	0f 84 ff 01 00 00                               	je     404660 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x2a0>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:174
  404461:	40 80 fd 7b                                     	cmp    bpl,0x7b
  404465:	0f 84 75 01 00 00                               	je     4045e0 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x220>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:179
  40446b:	89 e8                                           	mov    eax,ebp
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:182
  40446d:	44 0f be c5                                     	movsx  r8d,bpl
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:179
  404471:	83 e0 df                                        	and    eax,0xffffffdf
  404474:	3c 5d                                           	cmp    al,0x5d
  404476:	0f 84 8c 02 00 00                               	je     404708 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x348>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:182
  40447c:	4c 8b ab c0 00 00 00                            	mov    r13,QWORD PTR [rbx+0xc0]
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:933
  404483:	44 0f b6 f5                                     	movzx  r14d,bpl
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:181
  404487:	49 89 dc                                        	mov    r12,rbx
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:933
  40448a:	43 0f b6 84 35 39 01 00 00                      	movzx  eax,BYTE PTR [r13+r14*1+0x139]
  404493:	84 c0                                           	test   al,al
  404495:	0f 85 8d 02 00 00                               	jne    404728 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x368>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:935
  40449b:	49 8b 45 00                                     	mov    rax,QWORD PTR [r13+0x0]
  40449f:	48 8b 40 40                                     	mov    rax,QWORD PTR [rax+0x40]
  4044a3:	48 3d 90 30 40 00                               	cmp    rax,0x403090
  4044a9:	0f 85 41 02 00 00                               	jne    4046f0 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x330>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:936
  4044af:	40 84 ed                                        	test   bpl,bpl
  4044b2:	74 08                                           	je     4044bc <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0xfc>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:937
  4044b4:	43 88 ac 35 39 01 00 00                         	mov    BYTE PTR [r13+r14*1+0x139],bpl
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:183
  4044bc:	0f b6 03                                        	movzx  eax,BYTE PTR [rbx]
  4044bf:	84 c0                                           	test   al,al
  4044c1:	75 13                                           	jne    4044d6 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x116>
  4044c3:	eb 46                                           	jmp    40450b <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x14b>
  4044c5:	0f 1f 00                                        	nop    DWORD PTR [rax]
  4044c8:	41 0f b6 44 24 08                               	movzx  eax,BYTE PTR [r12+0x8]
  4044ce:	49 83 c4 08                                     	add    r12,0x8
  4044d2:	84 c0                                           	test   al,al
  4044d4:	74 35                                           	je     40450b <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x14b>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:184
  4044d6:	40 38 c5                                        	cmp    bpl,al
  4044d9:	75 ed                                           	jne    4044c8 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x108>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:186
  4044db:	41 8b 44 24 04                                  	mov    eax,DWORD PTR [r12+0x4]
  4044e0:	89 83 90 00 00 00                               	mov    DWORD PTR [rbx+0x90],eax
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:187
  4044e6:	eb 23                                           	jmp    40450b <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x14b>
  4044e8:	0f 1f 84 00 00 00 00 00                         	nop    DWORD PTR [rax+rax*1+0x0]
_ZNKSt8__detail12_ScannerBase10_M_is_ecmaEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.h:126
  4044f0:	8b 83 8c 00 00 00                               	mov    eax,DWORD PTR [rbx+0x8c]
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:126
  4044f6:	a8 10                                           	test   al,0x10
  4044f8:	75 6e                                           	jne    404568 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x1a8>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:158
  4044fa:	a8 02                                           	test   al,0x2
  4044fc:	0f 95 c0                                        	setne  al
  4044ff:	0f b6 c0                                        	movzx  eax,al
  404502:	83 c0 05                                        	add    eax,0x5
  404505:	89 83 90 00 00 00                               	mov    DWORD PTR [rbx+0x90],eax
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:196
  40450b:	5b                                              	pop    rbx
  40450c:	5d                                              	pop    rbp
  40450d:	41 5c                                           	pop    r12
  40450f:	41 5d                                           	pop    r13
  404511:	41 5e                                           	pop    r14
  404513:	c3                                              	ret
  404514:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:109
  404518:	48 8b 83 b0 00 00 00                            	mov    rax,QWORD PTR [rbx+0xb0]
  40451f:	48 3b 83 b8 00 00 00                            	cmp    rax,QWORD PTR [rbx+0xb8]
  404526:	0f 84 5b 02 00 00                               	je     404787 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x3c7>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:115
  40452c:	f7 83 8c 00 00 00 20 01 00 00                   	test   DWORD PTR [rbx+0x8c],0x120
  404536:	0f 84 c4 00 00 00                               	je     404600 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x240>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:117 (discriminator 2)
  40453c:	0f b6 10                                        	movzx  edx,BYTE PTR [rax]
  40453f:	8d 4a d8                                        	lea    ecx,[rdx-0x28]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:115 (discriminator 2)
  404542:	80 f9 01                                        	cmp    cl,0x1
  404545:	76 09                                           	jbe    404550 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x190>
  404547:	80 fa 7b                                        	cmp    dl,0x7b
  40454a:	0f 85 b0 00 00 00                               	jne    404600 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x240>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:122
  404550:	48 8d 50 01                                     	lea    rdx,[rax+0x1]
  404554:	48 89 93 b0 00 00 00                            	mov    QWORD PTR [rbx+0xb0],rdx
  40455b:	0f b6 28                                        	movzx  ebp,BYTE PTR [rax]
  40455e:	e9 e0 fe ff ff                                  	jmp    404443 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x83>
  404563:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:126 (discriminator 1)
  404568:	48 8b 93 b0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xb0]
  40456f:	80 3a 3f                                        	cmp    BYTE PTR [rdx],0x3f
  404572:	75 86                                           	jne    4044fa <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x13a>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:128
  404574:	48 8d 42 01                                     	lea    rax,[rdx+0x1]
  404578:	48 89 83 b0 00 00 00                            	mov    QWORD PTR [rbx+0xb0],rax
  40457f:	48 3b 83 b8 00 00 00                            	cmp    rax,QWORD PTR [rbx+0xb8]
  404586:	0f 84 0a 02 00 00                               	je     404796 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x3d6>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:133
  40458c:	0f b6 42 01                                     	movzx  eax,BYTE PTR [rdx+0x1]
  404590:	3c 3a                                           	cmp    al,0x3a
  404592:	0f 84 98 01 00 00                               	je     404730 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x370>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:138
  404598:	3c 3d                                           	cmp    al,0x3d
  40459a:	0f 84 aa 01 00 00                               	je     40474a <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x38a>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:144
  4045a0:	3c 21                                           	cmp    al,0x21
  4045a2:	0f 85 d0 01 00 00                               	jne    404778 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x3b8>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:146
  4045a8:	48 83 c2 02                                     	add    rdx,0x2
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:148
  4045ac:	48 8d bb c8 00 00 00                            	lea    rdi,[rbx+0xc8]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:147
  4045b3:	c7 83 90 00 00 00 07 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x7
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  4045bd:	41 b8 6e 00 00 00                               	mov    r8d,0x6e
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:146
  4045c3:	48 89 93 b0 00 00 00                            	mov    QWORD PTR [rbx+0xb0],rdx
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  4045ca:	48 8b 93 d0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd0]
  4045d1:	e9 dd 00 00 00                                  	jmp    4046b3 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x2f3>
  4045d6:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:176
  4045e0:	c7 83 88 00 00 00 01 00 00 00                   	mov    DWORD PTR [rbx+0x88],0x1
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:177
  4045ea:	c7 83 90 00 00 00 0c 00 00 00                   	mov    DWORD PTR [rbx+0x90],0xc
  4045f4:	e9 12 ff ff ff                                  	jmp    40450b <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x14b>
  4045f9:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:119
  404600:	48 8b bb f0 00 00 00                            	mov    rdi,QWORD PTR [rbx+0xf0]
  404607:	48 8b 83 e8 00 00 00                            	mov    rax,QWORD PTR [rbx+0xe8]
  40460e:	48 01 df                                        	add    rdi,rbx
  404611:	a8 01                                           	test   al,0x1
  404613:	74 08                                           	je     40461d <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x25d>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:119 (discriminator 1)
  404615:	48 8b 17                                        	mov    rdx,QWORD PTR [rdi]
  404618:	48 8b 44 02 ff                                  	mov    rax,QWORD PTR [rdx+rax*1-0x1]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:196 (discriminator 4)
  40461d:	5b                                              	pop    rbx
  40461e:	5d                                              	pop    rbp
  40461f:	41 5c                                           	pop    r12
  404621:	41 5d                                           	pop    r13
  404623:	41 5e                                           	pop    r14
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:119 (discriminator 4)
  404625:	ff e0                                           	jmp    rax
  404627:	66 0f 1f 84 00 00 00 00 00                      	nop    WORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:161
  404630:	c7 83 90 00 00 00 08 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x8
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:196
  40463a:	5b                                              	pop    rbx
  40463b:	5d                                              	pop    rbp
  40463c:	41 5c                                           	pop    r12
  40463e:	41 5d                                           	pop    r13
  404640:	41 5e                                           	pop    r14
  404642:	c3                                              	ret
  404643:	0f 1f 44 00 00                                  	nop    DWORD PTR [rax+rax*1+0x0]
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:935
  404648:	ba 20 00 00 00                                  	mov    edx,0x20
  40464d:	4c 89 e7                                        	mov    rdi,r12
  404650:	ff d1                                           	call   rcx
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:101
  404652:	0f be f0                                        	movsx  esi,al
  404655:	e9 be fd ff ff                                  	jmp    404418 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x58>
  40465a:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:165
  404660:	c6 83 a8 00 00 00 01                            	mov    BYTE PTR [rbx+0xa8],0x1
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:166
  404667:	48 8b 83 b0 00 00 00                            	mov    rax,QWORD PTR [rbx+0xb0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:164
  40466e:	c7 83 88 00 00 00 02 00 00 00                   	mov    DWORD PTR [rbx+0x88],0x2
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:166
  404678:	48 3b 83 b8 00 00 00                            	cmp    rax,QWORD PTR [rbx+0xb8]
  40467f:	74 05                                           	je     404686 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x2c6>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:166 (discriminator 1)
  404681:	80 38 5e                                        	cmp    BYTE PTR [rax],0x5e
  404684:	74 4a                                           	je     4046d0 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x310>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:172
  404686:	c7 83 90 00 00 00 09 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x9
  404690:	e9 76 fe ff ff                                  	jmp    40450b <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x14b>
  404695:	0f 1f 00                                        	nop    DWORD PTR [rax]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:103
  404698:	c7 83 90 00 00 00 01 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x1
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  4046a2:	48 8b 93 d0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd0]
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:104
  4046a9:	48 8d bb c8 00 00 00                            	lea    rdi,[rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  4046b0:	45 89 e8                                        	mov    r8d,r13d
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:196
  4046b3:	5b                                              	pop    rbx
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  4046b4:	b9 01 00 00 00                                  	mov    ecx,0x1
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:196
  4046b9:	5d                                              	pop    rbp
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  4046ba:	31 f6                                           	xor    esi,esi
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:196
  4046bc:	41 5c                                           	pop    r12
  4046be:	41 5d                                           	pop    r13
  4046c0:	41 5e                                           	pop    r14
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  4046c2:	e9 59 dc ff ff                                  	jmp    402320 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc@plt>
  4046c7:	66 0f 1f 84 00 00 00 00 00                      	nop    WORD PTR [rax+rax*1+0x0]
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:168
  4046d0:	c7 83 90 00 00 00 0a 00 00 00                   	mov    DWORD PTR [rbx+0x90],0xa
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:169
  4046da:	48 83 c0 01                                     	add    rax,0x1
  4046de:	48 89 83 b0 00 00 00                            	mov    QWORD PTR [rbx+0xb0],rax
  4046e5:	e9 21 fe ff ff                                  	jmp    40450b <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x14b>
  4046ea:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/locale_facets.h:935
  4046f0:	31 d2                                           	xor    edx,edx
  4046f2:	44 89 c6                                        	mov    esi,r8d
  4046f5:	4c 89 ef                                        	mov    rdi,r13
  4046f8:	ff d0                                           	call   rax
  4046fa:	89 c5                                           	mov    ebp,eax
  4046fc:	e9 ae fd ff ff                                  	jmp    4044af <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0xef>
  404701:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  404708:	48 8b 93 d0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd0]
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:194
  40470f:	48 8d bb c8 00 00 00                            	lea    rdi,[rbx+0xc8]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:193
  404716:	c7 83 90 00 00 00 01 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x1
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:901
  404720:	eb 91                                           	jmp    4046b3 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x2f3>
  404722:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
_ZNKSt5ctypeIcE6narrowEcc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:901
  404728:	89 c5                                           	mov    ebp,eax
  40472a:	e9 8d fd ff ff                                  	jmp    4044bc <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0xfc>
  40472f:	90                                              	nop
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:136
  404730:	c7 83 90 00 00 00 06 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x6
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:135
  40473a:	48 83 c2 02                                     	add    rdx,0x2
  40473e:	48 89 93 b0 00 00 00                            	mov    QWORD PTR [rbx+0xb0],rdx
  404745:	e9 c1 fd ff ff                                  	jmp    40450b <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x14b>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:140
  40474a:	48 83 c2 02                                     	add    rdx,0x2
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:142
  40474e:	48 8d bb c8 00 00 00                            	lea    rdi,[rbx+0xc8]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:141
  404755:	c7 83 90 00 00 00 07 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x7
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  40475f:	41 b8 70 00 00 00                               	mov    r8d,0x70
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:140
  404765:	48 89 93 b0 00 00 00                            	mov    QWORD PTR [rbx+0xb0],rdx
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  40476c:	48 8b 93 d0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd0]
  404773:	e9 3b ff ff ff                                  	jmp    4046b3 <_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv+0x2f3>
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:151
  404778:	be 20 41 41 00                                  	mov    esi,0x414120
  40477d:	bf 05 00 00 00                                  	mov    edi,0x5
  404782:	e8 d9 db ff ff                                  	call   402360 <_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:110
  404787:	be 38 40 41 00                                  	mov    esi,0x414038
  40478c:	bf 02 00 00 00                                  	mov    edi,0x2
  404791:	e8 ca db ff ff                                  	call   402360 <_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:129
  404796:	be e8 40 41 00                                  	mov    esi,0x4140e8
  40479b:	bf 05 00 00 00                                  	mov    edi,0x5
  4047a0:	e8 bb db ff ff                                  	call   402360 <_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc>
  4047a5:	66 2e 0f 1f 84 00 00 00 00 00                   	nop    WORD PTR cs:[rax+rax*1+0x0]
  4047af:	90                                              	nop

00000000004047b0 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv>:
_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:265
  4047b0:	41 57                                           	push   r15
  4047b2:	41 56                                           	push   r14
  4047b4:	41 55                                           	push   r13
  4047b6:	41 54                                           	push   r12
  4047b8:	55                                              	push   rbp
  4047b9:	53                                              	push   rbx
  4047ba:	48 83 ec 08                                     	sub    rsp,0x8
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:268
  4047be:	48 8b 87 b0 00 00 00                            	mov    rax,QWORD PTR [rdi+0xb0]
  4047c5:	48 8b 8f b8 00 00 00                            	mov    rcx,QWORD PTR [rdi+0xb8]
  4047cc:	48 39 c8                                        	cmp    rax,rcx
  4047cf:	0f 84 13 02 00 00                               	je     4049e8 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x238>
  4047d5:	48 89 fb                                        	mov    rbx,rdi
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:273
  4047d8:	48 8d 50 01                                     	lea    rdx,[rax+0x1]
_ZNKSt5ctypeIcE2isEtc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/x86_64-linux-gnu/bits/ctype_inline.h:44
  4047dc:	48 8b b3 c0 00 00 00                            	mov    rsi,QWORD PTR [rbx+0xc0]
_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:273
  4047e3:	48 89 97 b0 00 00 00                            	mov    QWORD PTR [rdi+0xb0],rdx
  4047ea:	0f b6 38                                        	movzx  edi,BYTE PTR [rax]
_ZNKSt5ctypeIcE2isEtc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/x86_64-linux-gnu/bits/ctype_inline.h:44
  4047ed:	48 8b 76 30                                     	mov    rsi,QWORD PTR [rsi+0x30]
_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:275
  4047f1:	f6 44 7e 01 08                                  	test   BYTE PTR [rsi+rdi*2+0x1],0x8
  4047f6:	0f 85 84 00 00 00                               	jne    404880 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0xd0>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:283
  4047fc:	40 80 ff 2c                                     	cmp    dil,0x2c
  404800:	74 5e                                           	je     404860 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0xb0>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:286
  404802:	f7 83 8c 00 00 00 20 01 00 00                   	test   DWORD PTR [rbx+0x8c],0x120
  40480c:	0f 84 9e 01 00 00                               	je     4049b0 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x200>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:288 (discriminator 1)
  404812:	48 39 d1                                        	cmp    rcx,rdx
  404815:	0f 84 be 01 00 00                               	je     4049d9 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x229>
  40481b:	40 80 ff 5c                                     	cmp    dil,0x5c
  40481f:	0f 85 b4 01 00 00                               	jne    4049d9 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x229>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:288 (discriminator 2)
  404825:	80 78 01 7d                                     	cmp    BYTE PTR [rax+0x1],0x7d
  404829:	0f 85 aa 01 00 00                               	jne    4049d9 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x229>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:290
  40482f:	c7 83 88 00 00 00 00 00 00 00                   	mov    DWORD PTR [rbx+0x88],0x0
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:292
  404839:	48 83 c0 02                                     	add    rax,0x2
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:291
  40483d:	c7 83 90 00 00 00 0d 00 00 00                   	mov    DWORD PTR [rbx+0x90],0xd
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:292
  404847:	48 89 83 b0 00 00 00                            	mov    QWORD PTR [rbx+0xb0],rax
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:306
  40484e:	48 83 c4 08                                     	add    rsp,0x8
  404852:	5b                                              	pop    rbx
  404853:	5d                                              	pop    rbp
  404854:	41 5c                                           	pop    r12
  404856:	41 5d                                           	pop    r13
  404858:	41 5e                                           	pop    r14
  40485a:	41 5f                                           	pop    r15
  40485c:	c3                                              	ret
  40485d:	0f 1f 00                                        	nop    DWORD PTR [rax]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:284
  404860:	c7 83 90 00 00 00 19 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x19
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:306
  40486a:	48 83 c4 08                                     	add    rsp,0x8
  40486e:	5b                                              	pop    rbx
  40486f:	5d                                              	pop    rbp
  404870:	41 5c                                           	pop    r12
  404872:	41 5d                                           	pop    r13
  404874:	41 5e                                           	pop    r14
  404876:	41 5f                                           	pop    r15
  404878:	c3                                              	ret
  404879:	0f 1f 80 00 00 00 00                            	nop    DWORD PTR [rax+0x0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:277
  404880:	c7 83 90 00 00 00 1a 00 00 00                   	mov    DWORD PTR [rbx+0x90],0x1a
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  40488a:	44 0f be c7                                     	movsx  r8d,dil
  40488e:	b9 01 00 00 00                                  	mov    ecx,0x1
  404893:	31 f6                                           	xor    esi,esi
  404895:	48 8b 93 d0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd0]
_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:278
  40489c:	4c 8d ab c8 00 00 00                            	lea    r13,[rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1453
  4048a3:	4c 89 ef                                        	mov    rdi,r13
  4048a6:	e8 75 da ff ff                                  	call   402320 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc@plt>
_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:279
  4048ab:	48 8b 83 b0 00 00 00                            	mov    rax,QWORD PTR [rbx+0xb0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:280
  4048b2:	48 3b 83 b8 00 00 00                            	cmp    rax,QWORD PTR [rbx+0xb8]
  4048b9:	74 93                                           	je     40484e <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x9e>
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:203
  4048bb:	4c 8d b3 d8 00 00 00                            	lea    r14,[rbx+0xd8]
  4048c2:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
_ZNKSt5ctypeIcE2isEtc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/x86_64-linux-gnu/bits/ctype_inline.h:44 (discriminator 1)
  4048c8:	48 8b 93 c0 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xc0]
  4048cf:	0f b6 08                                        	movzx  ecx,BYTE PTR [rax]
  4048d2:	48 8b 52 30                                     	mov    rdx,QWORD PTR [rdx+0x30]
_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:280 (discriminator 1)
  4048d6:	f6 44 4a 01 08                                  	test   BYTE PTR [rdx+rcx*2+0x1],0x8
  4048db:	0f 84 6d ff ff ff                               	je     40484e <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x9e>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:281
  4048e1:	48 8d 50 01                                     	lea    rdx,[rax+0x1]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:902
  4048e5:	48 8b ab d0 00 00 00                            	mov    rbp,QWORD PTR [rbx+0xd0]
_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:281
  4048ec:	48 89 93 b0 00 00 00                            	mov    QWORD PTR [rbx+0xb0],rdx
  4048f3:	44 0f b6 38                                     	movzx  r15d,BYTE PTR [rax]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:187
  4048f7:	48 8b 83 c8 00 00 00                            	mov    rax,QWORD PTR [rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1323
  4048fe:	4c 8d 65 01                                     	lea    r12,[rbp+0x1]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8capacityEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:966
  404902:	4c 39 f0                                        	cmp    rax,r14
  404905:	0f 84 95 00 00 00                               	je     4049a0 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x1f0>
  40490b:	48 8b 93 d8 00 00 00                            	mov    rdx,QWORD PTR [rbx+0xd8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1323
  404912:	49 39 d4                                        	cmp    r12,rdx
  404915:	77 39                                           	ja     404950 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x1a0>
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  404917:	44 88 3c 28                                     	mov    BYTE PTR [rax+rbp*1],r15b
  40491b:	48 8b 83 c8 00 00 00                            	mov    rax,QWORD PTR [rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:183
  404922:	4c 89 a3 d0 00 00 00                            	mov    QWORD PTR [rbx+0xd0],r12
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  404929:	c6 44 28 01 00                                  	mov    BYTE PTR [rax+rbp*1+0x1],0x0
_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:279
  40492e:	48 8b 83 b0 00 00 00                            	mov    rax,QWORD PTR [rbx+0xb0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:280
  404935:	48 3b 83 b8 00 00 00                            	cmp    rax,QWORD PTR [rbx+0xb8]
  40493c:	75 8a                                           	jne    4048c8 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x118>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:306
  40493e:	48 83 c4 08                                     	add    rsp,0x8
  404942:	5b                                              	pop    rbx
  404943:	5d                                              	pop    rbp
  404944:	41 5c                                           	pop    r12
  404946:	41 5d                                           	pop    r13
  404948:	41 5e                                           	pop    r14
  40494a:	41 5f                                           	pop    r15
  40494c:	c3                                              	ret
  40494d:	0f 1f 00                                        	nop    DWORD PTR [rax]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:1324
  404950:	41 b8 01 00 00 00                               	mov    r8d,0x1
  404956:	31 c9                                           	xor    ecx,ecx
  404958:	31 d2                                           	xor    edx,edx
  40495a:	48 89 ee                                        	mov    rsi,rbp
  40495d:	4c 89 ef                                        	mov    rdi,r13
  404960:	e8 9b d9 ff ff                                  	call   402300 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm@plt>
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  404965:	48 8b 83 c8 00 00 00                            	mov    rax,QWORD PTR [rbx+0xc8]
  40496c:	44 88 3c 28                                     	mov    BYTE PTR [rax+rbp*1],r15b
  404970:	48 8b 83 c8 00 00 00                            	mov    rax,QWORD PTR [rbx+0xc8]
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:183
  404977:	4c 89 a3 d0 00 00 00                            	mov    QWORD PTR [rbx+0xd0],r12
_ZNSt11char_traitsIcE6assignERcRKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/char_traits.h:322
  40497e:	c6 44 28 01 00                                  	mov    BYTE PTR [rax+rbp*1+0x1],0x0
_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:279
  404983:	48 8b 83 b0 00 00 00                            	mov    rax,QWORD PTR [rbx+0xb0]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:280
  40498a:	48 39 83 b8 00 00 00                            	cmp    QWORD PTR [rbx+0xb8],rax
  404991:	0f 85 31 ff ff ff                               	jne    4048c8 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x118>
  404997:	e9 b2 fe ff ff                                  	jmp    40484e <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x9e>
  40499c:	0f 1f 40 00                                     	nop    DWORD PTR [rax+0x0]
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8capacityEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/basic_string.h:966
  4049a0:	ba 0f 00 00 00                                  	mov    edx,0xf
  4049a5:	e9 68 ff ff ff                                  	jmp    404912 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x162>
  4049aa:	66 0f 1f 44 00 00                               	nop    WORD PTR [rax+rax*1+0x0]
_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:298
  4049b0:	40 80 ff 7d                                     	cmp    dil,0x7d
  4049b4:	75 23                                           	jne    4049d9 <_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv+0x229>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:300
  4049b6:	c7 83 88 00 00 00 00 00 00 00                   	mov    DWORD PTR [rbx+0x88],0x0
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:301
  4049c0:	c7 83 90 00 00 00 0d 00 00 00                   	mov    DWORD PTR [rbx+0x90],0xd
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:306
  4049ca:	48 83 c4 08                                     	add    rsp,0x8
  4049ce:	5b                                              	pop    rbx
  4049cf:	5d                                              	pop    rbp
  4049d0:	41 5c                                           	pop    r12
  4049d2:	41 5d                                           	pop    r13
  4049d4:	41 5e                                           	pop    r14
  4049d6:	41 5f                                           	pop    r15
  4049d8:	c3                                              	ret
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:295
  4049d9:	be 80 41 41 00                                  	mov    esi,0x414180
  4049de:	bf 07 00 00 00                                  	mov    edi,0x7
  4049e3:	e8 78 d9 ff ff                                  	call   402360 <_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc>
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/regex_scanner.tcc:269
  4049e8:	be 48 41 41 00                                  	mov    esi,0x414148
  4049ed:	bf 06 00 00 00                                  	mov    edi,0x6
  4049f2:	e8 69 d9 ff ff                                  	call   402360 <_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc>
  4049f7:	66 0f 1f 84 00 00 00 00 00                      	nop    WORD PTR [rax+rax*1+0x0]

0000000000404a00 <_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_>:
_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/vector.tcc:426
  404a00:	41 57                                           	push   r15
  404a02:	41 56                                           	push   r14
  404a04:	41 55                                           	push   r13
  404a06:	41 54                                           	push   r12
  404a08:	55                                              	push   rbp
  404a09:	53                                              	push   rbx
_ZNKSt6vectorImSaImEE12_M_check_lenEmPKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:1758
  404a0a:	48 bb ff ff ff ff ff ff ff 0f                   	movabs rbx,0xfffffffffffffff
_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/vector.tcc:426
  404a14:	48 83 ec 18                                     	sub    rsp,0x18
  404a18:	4c 8b 77 08                                     	mov    r14,QWORD PTR [rdi+0x8]
  404a1c:	4c 8b 2f                                        	mov    r13,QWORD PTR [rdi]
_ZNKSt6vectorImSaImEE4sizeEv():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:919
  404a1f:	4c 89 f0                                        	mov    rax,r14
  404a22:	4c 29 e8                                        	sub    rax,r13
  404a25:	48 c1 f8 03                                     	sar    rax,0x3
_ZNKSt6vectorImSaImEE12_M_check_lenEmPKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:1758
  404a29:	48 39 d8                                        	cmp    rax,rbx
  404a2c:	0f 84 2d 01 00 00                               	je     404b5f <_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_+0x15f>
  404a32:	48 85 c0                                        	test   rax,rax
  404a35:	49 89 d7                                        	mov    r15,rdx
  404a38:	ba 01 00 00 00                                  	mov    edx,0x1
  404a3d:	48 89 fd                                        	mov    rbp,rdi
  404a40:	48 0f 45 d0                                     	cmovne rdx,rax
  404a44:	31 c9                                           	xor    ecx,ecx
  404a46:	49 89 f4                                        	mov    r12,rsi
  404a49:	48 01 d0                                        	add    rax,rdx
_ZN9__gnu_cxxmiIPmSt6vectorImSaImEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_iterator.h:1180
  404a4c:	48 89 f2                                        	mov    rdx,rsi
  404a4f:	0f 92 c1                                        	setb   cl
  404a52:	4c 29 ea                                        	sub    rdx,r13
_ZNKSt6vectorImSaImEE12_M_check_lenEmPKc():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:1762
  404a55:	48 85 c9                                        	test   rcx,rcx
  404a58:	0f 85 f2 00 00 00                               	jne    404b50 <_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_+0x150>
_ZNSt12_Vector_baseImSaImEE11_M_allocateEm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:346
  404a5e:	48 85 c0                                        	test   rax,rax
  404a61:	75 5d                                           	jne    404ac0 <_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_+0xc0>
  404a63:	31 db                                           	xor    ebx,ebx
_ZN9__gnu_cxx13new_allocatorImE9constructImJRKmEEEvPT_DpOT0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:150
  404a65:	49 8b 07                                        	mov    rax,QWORD PTR [r15]
_ZSt14__relocate_a_1ImmENSt9enable_ifIXsrSt24__is_bitwise_relocatableIT_vE5valueEPS2_E4typeES4_S4_S4_RSaIT0_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_uninitialized.h:991
  404a68:	4d 89 f0                                        	mov    r8,r14
_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/vector.tcc:464
  404a6b:	4c 8d 4c 11 08                                  	lea    r9,[rcx+rdx*1+0x8]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/vector.tcc:501
  404a70:	4c 8b 75 10                                     	mov    r14,QWORD PTR [rbp+0x10]
_ZSt14__relocate_a_1ImmENSt9enable_ifIXsrSt24__is_bitwise_relocatableIT_vE5valueEPS2_E4typeES4_S4_S4_RSaIT0_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_uninitialized.h:991
  404a74:	4d 29 e0                                        	sub    r8,r12
_ZN9__gnu_cxx13new_allocatorImE9constructImJRKmEEEvPT_DpOT0_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/ext/new_allocator.h:150
  404a77:	48 89 04 11                                     	mov    QWORD PTR [rcx+rdx*1],rax
_ZSt14__relocate_a_1ImmENSt9enable_ifIXsrSt24__is_bitwise_relocatableIT_vE5valueEPS2_E4typeES4_S4_S4_RSaIT0_E():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_uninitialized.h:994
  404a7b:	4f 8d 3c 01                                     	lea    r15,[r9+r8*1]
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_uninitialized.h:992
  404a7f:	48 85 d2                                        	test   rdx,rdx
  404a82:	7f 6c                                           	jg     404af0 <_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_+0xf0>
  404a84:	4d 85 c0                                        	test   r8,r8
  404a87:	0f 8f a3 00 00 00                               	jg     404b30 <_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_+0x130>
_ZNSt12_Vector_baseImSaImEE13_M_deallocateEPmm():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/stl_vector.h:353
  404a8d:	4d 85 ed                                        	test   r13,r13
  404a90:	0f 85 7f 00 00 00                               	jne    404b15 <_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_+0x115>
_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_():
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/vector.tcc:502
  404a96:	66 49 0f 6e cf                                  	movq   xmm1,r15
  404a9b:	66 48 0f 6e c1                                  	movq   xmm0,rcx
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/vector.tcc:504
  404aa0:	48 89 5d 10                                     	mov    QWORD PTR [rbp+0x10],rbx
/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/bits/vector.tcc:502
  404aa4:	66 0f 6c c1                                     	punpcklqdq xmm0,xmm1
