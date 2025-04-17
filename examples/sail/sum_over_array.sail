default Order dec

$include <prelude.sail>
$include <vector.sail>

// This is generic over all fixed-length vectors. `input` is not
// dynamically sized.
function sum_array forall 'n . (input : vector('n, int)) -> int = {
    // Type must be explicitly declared as `int`. If you leave
    // that out the type will be inferred as `int(0)` and it
    // can only contain the value 0.
    var total : int = 0;
    foreach (i from 0 to (length(input) - 1)) {
        // You can easily add `+=` operators - Sail has good
        // support for operator overloading - but it isn't
        // there by default.
        total = total + input[i];
    };
    total
}

// You can also use functional style lists, but these are
// not supported by all backends.
function sum_array_fp (input : list(int)) -> int =
    match input {
        // Empty list. List literals are with [| ... |].
        [| |] => 0,
        // Single element.
        [| el |] => el,
        // More than one element.
        head :: tail => head + sum_array_fp(tail),
    }

function main() -> unit = {
    print_endline(concat_str("Sum of [1, 2, 3] is ", dec_str(sum_array([1 : int, 2, 3]))));
    print_endline(concat_str("Sum of [1, 2, 3] is ", dec_str(sum_array_fp([| 1, 2, 3 |]))));
}
