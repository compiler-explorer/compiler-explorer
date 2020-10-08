# For best results compile with -d:danger --passC:"-march=native" 
proc sumArray(data: openArray[int]): int {.exportc.} = 
  for elem in data:
    result += elem
