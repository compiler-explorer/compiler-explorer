package main;

func MaxArray(x []float64, y []float64) {
  for i, c := range x {
    if y[i] > c { x[i] = y[i] }
  }
}
