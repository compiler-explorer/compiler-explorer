# For best results with compile with -d:danger --passC:"-march=native"
proc maxArray(x: var openArray[float], y: openArray[float]) {.exportc.} = 
  for i in 0 ..< x.len:
    if y[i] > x[i]:
      x[i] = y[i]
