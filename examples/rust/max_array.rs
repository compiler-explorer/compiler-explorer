pub fn max_array(x: &mut[int, ..65536], y: &[int, ..65536]) {
  for i in range(0u, 65536) {
    if y[i] > x[i] {
      x[i] = y[i];
    }
  }
}
