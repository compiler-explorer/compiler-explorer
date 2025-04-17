default Order dec

$include <prelude.sail>
$include <vector.sail>

// Bitfields are useful for registers. You can take an underlying bitvector
// and allow naming bits in them.

bitfield Status : bits(32) = {
    U : 31,
    // Not all bits need to have a corresponding field, and fields can overlap.
    A : 0,
    lowest_byte : 7 .. 0,
    upper_byte : 31 .. 24,
}

function main() -> unit = {
    // To create a new bitfield, use Mk_<name>.
    var status = Mk_Status(0x00000000);

    // You can update mutable variables (including `register`s) imperatively:
    status[lowest_byte] = status[upper_byte];

    // You can access the underlying bitvector via the special field `bits`.
    status.bits = 0x11223344;

    // You can also do functional-style immutable updates.
    let status = [
        status with
        U = 0b1,
        lowest_byte = 0xFF,
    ];

    print_endline(bits_str(status.bits));
}
