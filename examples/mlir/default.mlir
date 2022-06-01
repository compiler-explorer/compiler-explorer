func @affine_parallel_with_reductions_i64(%arg0: memref<3x3xi64>, %arg1: memref<3x3xi64>) -> (i64, i64) {
  %0:2 = affine.parallel (%kx, %ky) = (0, 0) to (2, 2) reduce ("addi", "muli") -> (i64, i64) {
            %1 = affine.load %arg0[%kx, %ky] : memref<3x3xi64>
            %2 = affine.load %arg1[%kx, %ky] : memref<3x3xi64>
            %3 = arith.muli %1, %2 : i64
            %4 = arith.addi %1, %2 : i64
            affine.yield %3, %4 : i64, i64
          }
  return %0#0, %0#1 : i64, i64
}
