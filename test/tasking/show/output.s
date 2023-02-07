---------- Section dump ----------

.text.example.main
00000000 20 20        constT1&Max<double>(constT1&,constT1&):sub.a       sp,#0x20
00000002 6d 00 00 00                   call        0x2
00000006 da 27                         mov         d15,#0x27
00000008 74 af                         st.w        [sp],d15
0000000a da 14                         mov         d15,#0x14
0000000c 78 01                         st.w        [sp]0x4,d15
0000000e 49 af 00 0a                   lea         a15,[sp]0x0
00000012 49 a2 04 0a                   lea         a2,[sp]0x4
00000016 54 ff                         ld.w        d15,[a15]
00000018 ee 0c                         jnz         d15,0x30
0000001a da 00                         mov         d15,#0x0
0000001c 53 1f 22 00                   mul         d0,d15,#0x21
00000020 54 21                         ld.w        d1,[a2]
00000022 da 09                         mov         d15,#0x9
00000024 03 f1 0a f0                   madd        d15,d0,d1,d15
00000028 78 06                         st.w        [sp]0x18,d15
0000002a 49 af 18 0a                   lea         a15,[sp]0x18
0000002e 3c 01                         jg          0x30
00000030 54 f4                         ld.w        d4,[a15]
00000032 85 00 00 04                   ld.d        d0/d1,constT1&Max<double>(constT1&,constT1&)
00000036 89 a0 48 09                   st.d        [sp]0x8,d0/d1
0000003a 85 00 00 04                   ld.d        d0/d1,constT1&Max<double>(constT1&,constT1&)
0000003e 89 a0 50 09                   st.d        [sp]0x10,d0/d1
00000042 09 a0 48 09                   ld.d        d0/d1,[sp]0x8
00000046 d2 02                         mov         d2/d3,#0x0
00000048 4b 20 02 f0                   cmp.df      d15,d0/d1,d2/d3
0000004c 6f 1f 13 00                   jz.t        d15:0x1,0x72
00000050 09 a0 48 09                   ld.d        d0/d1,[sp]0x8
00000054 85 02 00 04                   ld.d        d2/d3,constT1&Max<double>(constT1&,constT1&)
00000058 09 a6 50 09                   ld.d        d6/d7,[sp]0x10
0000005c 85 08 00 04                   ld.d        d8/d9,constT1&Max<double>(constT1&,constT1&)
00000060 4b 86 42 60                   mul.df      d6/d7,d6/d7,d8/d9
00000064 6b 20 62 06                   madd.df     d0/d1,d6/d7,d0/d1,d2/d3
00000068 89 a0 58 09                   st.d        [sp]0x18,d0/d1
0000006c 49 af 18 0a                   lea         a15,[sp]0x18
00000070 3c 01                         jg          0x72
00000072 09 f0 40 09                   ld.d        d0/d1,[a15]0x0
00000076 4b 04 42 21                   itodf       d2/d3,d4
0000007a 6b 00 22 02                   add.df      d0/d1,d2/d3,d0/d1
0000007e 4b 00 32 21                   dftoiz      d2,d0/d1
00000082 3c 01                         jg          0x84
00000084 00 90                         ret

.text.example._Z3MaxIiERKT_S2_S2_
00000000 20 08        constT1&Max<double>(constT1&,constT1&):sub.a       sp,#0x8
00000002 54 4f                         ld.w        d15,[a4]
00000004 ee 0c                         jnz         d15,0x1c
00000006 da 00                         mov         d15,#0x0
00000008 53 1f 22 00                   mul         d0,d15,#0x21
0000000c 54 51                         ld.w        d1,[a5]
0000000e da 09                         mov         d15,#0x9
00000010 03 f1 0a f0                   madd        d15,d0,d1,d15
00000014 74 af                         st.w        [sp],d15
00000016 49 a2 00 0a                   lea         a2,[sp]0x0
0000001a 3c 01                         jg          0x1c
0000001c 00 90                         ret

.text.example._Z3MaxIdERKT_S2_S2_
00000000 20 08        constT1&Max<double>(constT1&,constT1&):sub.a       sp,#0x8
00000002 09 40 40 09                   ld.d        d0/d1,[a4]0x0
00000006 d2 02                         mov         d2/d3,#0x0
00000008 4b 20 02 f0                   cmp.df      d15,d0/d1,d2/d3
0000000c 6f 1f 13 00                   jz.t        d15:0x1,0x32
00000010 09 40 40 09                   ld.d        d0/d1,[a4]0x0
00000014 85 02 00 04                   ld.d        d2/d3,constT1&Max<double>(constT1&,constT1&)
00000018 09 54 40 09                   ld.d        d4/d5,[a5]0x0
0000001c 85 06 00 04                   ld.d        d6/d7,constT1&Max<double>(constT1&,constT1&)
00000020 4b 64 42 40                   mul.df      d4/d5,d4/d5,d6/d7
00000024 6b 20 62 04                   madd.df     d0/d1,d4/d5,d0/d1,d2/d3
00000028 89 a0 40 09                   st.d        [sp]0x0,d0/d1
0000002c 49 a2 00 0a                   lea         a2,[sp]0x0
00000030 3c 01                         jg          0x32
00000032 00 90                         ret