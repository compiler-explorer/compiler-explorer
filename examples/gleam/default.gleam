import gleam/io
import gleam/int

pub fn fibonacci(n: Int) -> Int {
  case n {
    0 -> 0
    1 -> 1
    _ -> fibonacci(n - 1) + fibonacci(n - 2)
  }
}

pub fn main() {
  let result = fibonacci(10)
  io.println("fib(10) = " <> int.to_string(result))
}
