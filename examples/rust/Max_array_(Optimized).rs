// Compile with -C opt-level=3 -C target-cpu=native to see autovectorization

#[repr(align(64))]
pub struct Aligned<T: ?Sized>(T);

pub fn max_array(x: &mut Aligned<[f64; 65536]>, y: &Aligned<[f64; 65536]>) {
    for (x, y) in x.0.iter_mut().zip(y.0.iter_mut()) {
        *x = if *y > *x { *y } else { *x };
    }
}
