// Compile with -C opt-level=3 -C target-cpu=native to see autovectorization

#[repr(align(64))]
pub struct Aligned<T: ?Sized>(T);

pub fn sum_array(input: &Aligned<[i32]>) -> i32 {
    input.0.iter().sum()
}
