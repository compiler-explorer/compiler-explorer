# For best results compile with -d:danger
proc sumArray(data: seq[int]): int {.exportc.} = 
  for elem in data:
    result += elem
