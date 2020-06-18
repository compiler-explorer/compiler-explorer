        .text
        .align  2
        .weakext        issue1989
$LFB0:
        .ent    issue1989
        .type   issue1989, @function
issue1989:
        .frame  r1,0,r15            # vars= 0, regs= 0, args= 0
        .mask   0x00000000
        addk    r6,r5,r6
        cmp     r18,r6,r5
        bltid   r18,$L2
        addik   r3,r0,1       # 0x1
        addk    r3,r0,r0
$L2:
        rtsd    r15,8

        andi    r3,r3,1 #and1
        .end    issue1989
$LFE0:
$Lfe1:
        .size   issue1989,$Lfe1-issue1989
        .section        .note.GNU-stack,"",@progbits
