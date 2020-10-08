# For best results compile with -d:danger
proc maxArray(x: var seq[float], y: seq[float]) {.exportc.} = 
  for i in 0 ..< x.len:
    if y[i] > x[i]:
      x[i] = y[i]
