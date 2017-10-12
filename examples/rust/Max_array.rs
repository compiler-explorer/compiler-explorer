pub fn max_array(x: &mut[f64; 65536], y: &[f64; 65536]) {
  for i in 0..65536 {
    x[i] = if y[i] > x[i] { y[i] } else { x[i] };
  }
}
