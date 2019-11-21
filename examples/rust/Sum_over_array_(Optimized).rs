#![feature(core_intrinsics)]
// Requires the use of the nightly rust
// Compile with -O

pub fn sum_array_loop(input: &[i32]) -> i32 {
  unsafe {
    std::intrinsics::assume(input.as_ptr() as usize % 64 == 0);
  }

  let mut sum: i32 = 0;

  for i in 0..input.len() {
    sum += input[i];
  }

  sum
}

pub fn sum_array_iterator(input: &[i32]) -> i32 {
  unsafe {
    std::intrinsics::assume(input.as_ptr() as usize % 64 == 0);
  }

  input.iter().sum()
}
