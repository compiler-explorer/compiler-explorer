# Type your code here, or load an example.
square : I32 -> I32
square = |num| num * num

main! = |_| Err(Exit(square(5).to_i8_wrap()))
