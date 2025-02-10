script;

fn square(x: u64) -> u64 {
    x * x
}

fn main() {
    let x = square(5);
    assert(x == 25);
}