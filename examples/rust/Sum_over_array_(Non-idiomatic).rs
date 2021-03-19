// Compile with -C opt-level=3 -C target-cpu=native to see autovectorization
// assumes input's length is a multiple of 64
pub fn sum_array(input: &[i32]) -> i32 {
    let len = input.len();
    let input = input.as_ptr();

    if len & 63 != 0 {
        unsafe { std::hint::unreachable_unchecked() }
    }

    let mut sum = 0;
    for i in 0..len {
        sum += unsafe { *input.add(i) };
    }

    sum
}
