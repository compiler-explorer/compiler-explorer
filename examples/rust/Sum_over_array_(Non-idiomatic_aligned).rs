// Compile with -C opt-level=3 -C target-cpu=native to see autovectorization

#[repr(align(64))]
pub struct Aligned<T: ?Sized>(T);

// assumes input is aligned on 64-byte boundary and that
// input's length is a multiple of 64.
pub fn sum_array(input: &Aligned<[i32]>) -> i32 {
    let len = input.0.len();
    let input = input.0.as_ptr();

    if len & 63 != 0 {
        unsafe { std::hint::unreachable_unchecked() }
    }

    let mut sum = 0;
    for i in 0..len {
        sum += unsafe { *input.add(i) };
    }

    sum
}
