default Order dec

$include <prelude.sail>
$include <result.sail>
$include <vector.sail>

// Basic types are:

// Any integer, limited only by your computer's memory!
let a : int = 1000000000000000000000000000000000000000000000000000000000000000

// Any natural number (non-negative integer).
let b : nat = 1000000000000000000000000000000000000000000000000000000000000000

// An inclusive range of integers.
let c : range(1, 6) = 3

// Strings. Sail only has very basic string support, mainly for logging and
// assembly/disassembly support.
let d : string = "hello"

// Boolean
let e : bool = false

// Unit (like an empty tuple; roughly equivalent to 'void').
let f : unit = ()

// Bit vector. Note that bit-vector literals are written in hex or binary
// and leading zeros are significant! 0xF is a 4-bit bitvector, but 0x0F is
// an 8-bit bitvector. Try changing this to 0xF or 0b1010101.
let g : bits(8) = 0x0F

// More complex types are:

// Standard functional option and result types.
let h : option(int) = Some(5)
let i : option(nat) = None()

let j : result(int, unit) = Ok(5)
let k : result(int, unit) = Err(())

// Fixed-length vectors.
let l : vector(4, int) = [1, 2, 3, 4]

// Dynamically sized linked lists.
let m : list(int) = [|1, 2, 3, 4|]

// Custom enum, union and struct types are supported.

// Enums are not namespaced; so you refer to `Red` directly, not `Primary::Red`.
enum Primary = { Red, Green, Blue }

// Unions are tagged (same as `enum` in Rust), and decoded using `match`.
union Instruction = {
    FullSize   : bits(32),
    Compressed : bits(16),
}

struct ExecutionStep = {
    pc          : bits(32),
    instruction : Instruction,
    retired     : bool,
}

function main() -> unit = ()
