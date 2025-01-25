// This example introduces most basic features of Sail.
/* Comments are C-style. /* Blocks comments can be nested. */ */

// Sail supports configurable indexing endianness. Dec is the most common
// option. It must be declared globally.
default Order dec

// C-style includes are supported. `prelude.sail` is from the Sail standard
// library and includes some common things.
$include <prelude.sail>

// Global mutable variables are introduced with `register`.
// They must include a type (int) and can include an initialiser (= 0).
// `int` is the type for all integers (any size, like in Python).
// Try changing this to 10000000000000000000000000000000000000000000000000.
register my_global_int : int = 0

// `nat` is the type for natural numbers (>= 0). Try changing the 3 to -3.
register not_negative : nat = 3

// There's also a `range(M, N)` type for integers in [M, N] (inclusive).
// Try changing this to 0
register month : range(1, 12) = 1

// Immutable globals variables are declared with let.
// Real numbers are supported, using libgmp (which represents them as
// arbitrary length integer fractions). Although supported, they are very
// rarely used in Sail code.
let pi : real = 3.141

// This is a simple function. Sail is expression-based like Rust, OCaml,
// Haskell, etc. Braces are not required for function bodies if they are
// a single expression.
function add_ints(a : int, b : int) -> int = a + b

// As usual in expression-based languages, if and match are also expressions.
function max1(a : int, b : int) -> int = if a > b then a else b

// You can use traditional imperative forms too but it isn't recommended.
function max2(a : int, b : int) -> int = {
    if a > b then {
        return a;
    } else {
        return b;
    }
}

// It has standard ML/Rust style `match`.
function month_name(month : range(1, 12)) -> string =
    match month {
        1 => "Jan",
        2 => "Feb",
        // Alternate patterns are not yet supported.
        // 3 | 4 => "March or April",
        // _ is a wildcard.
        _ => "Whatever",
    }

// You can define a `main()` function which will be compiled to `zmain()` in C.
// The name mangling system for C is Z-encoding which basically double all
// z's and prepends a z. E.g. size -> zsizze
//
// Main returns `unit` which is like a 0-length tuple. It's used in places you
// might use `void` in C. Unlike `void` you can make a value of `unit` by writing
// `()`.
//
// Sail will optionally generate a C `main()` function that calls `zmain()`
// (this code).
function main() -> unit = {
    // The standard print function.
    print_endline("Hello world");

    // `let` introduces immutable variables.
    let a : int = 5;
    // They can be shadowed with the same name.
    let a : string = "5";
    // Type annotations are not always required - often they can be inferred.
    let a = 5;
    // Because they are immutable you cannot do this:
    a = 6;

    // Mutable variables use `var`.
    var a : int = 5;
    // This is now ok.
    a = 6;

    // One small gotcha is that the inferred type for 5 is int(5). So this
    var a = 5;
    // is equivalent to this
    var a : int(5) = 5;
    // And because its type is literally the number 5, you can't assign
    // any other numbers to it.
    // a = 6; // compile error
}

// Being aimed at ISA specifications, Sail has very good support for bit vectors.
function reverse_endianness(word : bits(16)) -> bits(16) =
    // @ means concatenation. Since we declared `default Order dec` all
    // of these indices and concatenations happen in Little Endian notational
    // order. This doesn't mean that the underlying data is Little Endian though.
    //
    // .. selects a bit range. This is inclusive. Currently there is no
    // half-open syntax (8 >.. 0) but this is planned.
    word[7 .. 0] @ word[15 .. 8]

// Sail has an advanced type system for bit vectors and integers which means
// they can almost always be checked at compile time.
//
// Try changing 7 to 8 here.
function lowest_byte(word : bits(16)) -> bits(8) = word[7 .. 0]

// Check the other examples for more features. Or see the Sail manual here:
// https://alasdair.github.io/manual.html
