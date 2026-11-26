def add (x y : Nat) : Nat := x + y

def main : IO Unit := do
    IO.println (add 1 2)
