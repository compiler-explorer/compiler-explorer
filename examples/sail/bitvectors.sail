default Order dec

$include <prelude.sail>
$include <vector.sail>

// Sail has excellent support for bit vectors. These have the type `bits(N)`.
let status : bits(32) = 0x11223344

// Bit vector literals are written 0x.. or 0b...
// Leading zeros are significant - the affect the size of the literal.
// This is ok:
register byte0 : bits(8) = 0x01
// This is an error:
// register byte1 : bits(8) = 0x1

function main() -> unit = {
    // @ means concatenation. With 'default Order dec' concatenation is written
    // in Big Endian order.
    assert(0x12 @ 0x34 == 0x1234);

    // Individual bits or ranges of bits can be accessed like this.
    // Note that the range is inclusive.
    assert(status[0] == bitzero);
    assert(status[7 .. 0] == 0x44);

    // You can update mutable bit vectors like this:
    byte0[3 .. 0] = 0b1100;
    // Or in immutable function style:
    let byte2 = [
        byte0 with
        3 .. 0 = 0xA,
    ];

    // You can convert a bit vector to a string with `bits_str()`. If it is
    // a multiple of 4 bits it will be printed in hex, otherwise binary.
    print_endline(bits_str(byte2[2 .. 0]));

    // Pattern matching also supports bit vectors.
    match byte2 {
        upper @ 0b010 => {
            print_endline(bits_str(upper));
        },
        0b10 @ lower => {
            print_endline(bits_str(lower));
        },
        _ => (),
    };
}
