func.func @ppow(%x: f64) -> f64 {
  %cst = arith.constant 1.000000e+00 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 10 : index
  %r = scf.for %iv = %c0 to %n step %c1 iter_args(%r_it = %cst) -> f64 {
    %r_next = arith.mulf %r_it, %x : f64
    scf.yield %r_next : f64
  }
  return %r : f64
}

func.func @dppow(%x: f64, %dr: f64) -> f64 {
  %r = enzyme.autodiff @ppow(%x, %dr) { activity=[#enzyme<activity enzyme_out>] } : (f64, f64) -> f64
  return %r : f64
}
