// Example code of a simple counter.
// CIRCT example code may not always work out of the box because the textual MLIR format is not always stable.
// The example tries to be compatible with the latest CIRCT version, using relatively stable IR.

hw.module @Counter(in %clock: !seq.clock, in %reset: i1, out count: i8) {
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %counter = seq.compreg %0, %clock reset %reset, %c0_i8  : i8
  %0 = comb.add %counter, %c1_i8 : i8
  hw.output %counter : i8
}
