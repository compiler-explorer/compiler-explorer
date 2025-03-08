default Order dec

$include <option.sail>
$include <string.sail>

// Sail has pattern matching similar to Rust, OCaml and other functional languages.
// A notable difference is that it doesn't support combining identical branches.

function suit_name(suit : range(0, 3)) -> string =
    match suit {
        0 => "hearts",
        1 => "diamonds",
        2 => "clubs",
        3 => "spaces",
    }

// _ is a wildcard

function map_add(a : option(int), b : option(int)) -> option(int) =
    match (a, b) {
        (Some(a), Some(b)) => Some(a + b),
        _ => None(),
    }

// Sail copies a minor footgun from other functional languages.

function ordinal(x : nat) -> string =
    match x {
        0 => "zeroth",
        1 => "first",
        2 => "second",
        3 => "third",
        other => concat_str(dec_str(other), "th"),
    }

// Here `other` creates a new variable bound to `x`.
// Unfortunately it can catch you out. The following code appears correct and
// will compile (with warnings), but actually the function always returns 2.
// Rust and OCaml also suffer from this flaw.

let ONE = 1
let TWO = 2
let THREE = 3

function add_one(x : range(1, 3)) -> range(2, 4) =
    match x {
        ONE => 2,
        TWO => 3,
        THREE => 4,
    }

function main() -> unit = ()
