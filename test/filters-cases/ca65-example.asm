ca65 V2.19 - Git f166ca8
Main file   : /tmp/ca65-test.s
Current file: /tmp/ca65-test.s

000000r 1               .segment "CODE"
000000r 1               
000000r 1               ; Simple 6502 test with macros
000000r 1               .macro inc16 addr
000000r 1                   inc addr
000000r 1                   bne :+
000000r 1                   inc addr+1
000000r 1               :
000000r 1               .endmacro
000000r 1               
000000r 1               .proc main
000000r 1  A9 00            lda #$00
000002r 1  8D 00 D0         sta $D000
000005r 1                   inc16 counter
000005r 0  EE rr rr     >  inc addr
000008r 0  D0 03        >  bne :++
00000Ar 0  EE rr rr     >  inc addr+1
00000Dr 0               > :
00000Dr 1  60               rts
00000Er 1               .endproc
00000Er 1               
00000Er 1               .segment "BSS"
000000r 1  xx xx        counter: .res 2
000000r 1               
