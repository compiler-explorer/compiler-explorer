pub fn max_array(x: &mut [f64; 65536], y: &[f64; 65536]) {
    for (x, y) in x.iter_mut().zip(y.iter()) {
        *x = if *y > *x { *y } else { *x };
    }
}
