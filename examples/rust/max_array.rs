pub fn max_array(x: &mut[i32; 65536], y: &[i32; 65536]) {
  for i in (0..65536) {
    if y[i] > x[i] {
      x[i] = y[i];
    }
  }
}
