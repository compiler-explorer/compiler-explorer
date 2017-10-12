#![feature(core_intrinsics)]
// Requires the use of the nightly rust
// Compile with -O
pub fn max_array(x: &mut[f64; 65536], y: &[f64; 65536]) {
  unsafe {
    std::intrinsics::assume(x.as_ptr() as usize % 64 == 0);
    std::intrinsics::assume(y.as_ptr() as usize % 64 == 0);
  }
  for i in 0..65536 {
    x[i] = if y[i] > x[i] { y[i] } else { x[i] };
  }
}
