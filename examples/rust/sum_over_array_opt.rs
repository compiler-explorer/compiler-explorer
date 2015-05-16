#![feature(core)]
// Requires the use of the nightly rust
// Compile with -O
pub fn sum_array(x: &[i32]) -> i32 {
  unsafe {
    std::intrinsics::assume(x.as_ptr() as usize % 64 == 0);   
  }
  x.iter().fold(0, |sum, next| sum + *next)
}
