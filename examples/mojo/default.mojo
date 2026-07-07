@export
def multiply(x: Int, y: Int) abi("C") -> Int:
    return x * y
