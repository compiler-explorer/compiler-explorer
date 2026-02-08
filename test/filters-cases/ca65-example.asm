000000r 1               .segment "CODE"
000000r 1               
000000r 1               ; Simple 6502 program
000000r 1  A9 00           lda #$00
000002r 1  8D 00 D0        sta $D000
000005r 1               
000005r 1               .macro inc16 addr
000005r 1                   inc addr
000005r 1                   bne :+
000005r 1                   inc addr+1
000005r 1               :
000005r 1               .endmacro
000005r 1               
000005r 1               counter: .word $0000
000005r 2                   inc counter
000008r 2  D0 02           bne :+
00000Ar 2                   inc counter+1
00000Dr 2               :
00000Dr 1               
00000Dr 1  60              rts
