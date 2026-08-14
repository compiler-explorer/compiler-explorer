# export + abi("C"): keeps square from being stripped as unused, uses C calling
# convention (no unbound params), and gives it a clean mangled name.
@export
def square(i: Int32) abi("C") -> Int32:
    return i * i