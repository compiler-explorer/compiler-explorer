Instruction set:          Arm64
Instruction set features: a53,crc,-lse,-fp16,-dotprod,-sve
Compiler filter:          speed


void Square.<init>() [4 bytes]
    0x00001010    ret

int Square.square(int) [8 bytes]
    0x00001020    mul w0, w1, w1
    0x00001024    ret

