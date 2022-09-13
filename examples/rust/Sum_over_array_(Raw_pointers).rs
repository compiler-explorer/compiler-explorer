// Compile with -C opt-level=3 -C target-cpu=native to see autovectorization

// assumes input is aligned on 64-byte boundary and that
// input's length is a multiple of 64.
pub fn sum_array(input: &[i32]) -> i32 {
    if input.len() & 63 != 0 {
        unsafe { std::hint::unreachable_unchecked() }
    }

    (0..input.len())
        .map(|i| unsafe { *input.as_ptr().add(i) })
        .sum()
}
