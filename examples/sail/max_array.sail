default Order dec

$include <prelude.sail>
$include <vector.sail>

// This is generic over all fixed-length vectors. They are not dynamically sized.
function max_array forall 'n . (x : vector('n, int), y : vector('n, int)) -> vector('n, int) = {
    // vector_init() copies the value to all elements.
    var m : vector('n, int) = vector_init(0);
    foreach (i from 0 to ('n - 1)) {
        m[i] = if x[i] > y[i] then x[i] else y[i];
    };
    m
}

function main() -> unit = {
    let m = max_array([1, 2, 3], [3, 2, 1]);
    foreach (i from 0 to (length(m) - 1)) {
        print_endline(dec_str(m[i]));
    }
}
