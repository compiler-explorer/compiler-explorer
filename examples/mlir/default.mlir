func @multiple_conversion_casts(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %inputs:2 = builtin.unrealized_conversion_cast %arg0, %arg1 : i32, i32 to i64, i64
  %outputs:2 = builtin.unrealized_conversion_cast %inputs#0, %inputs#1 : i64, i64 to i32, i32
  return %outputs#0, %outputs#1 : i32, i32
}
