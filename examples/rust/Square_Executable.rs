use std::env;

const fn square(num: i32) -> i32 {
    num * num
}

pub fn main() {
    match env::args().nth(1).map(|r| r.parse::<i32>()) {
        Some(Ok(r)) => {
            println!("{}", square(r))
        }
        _ => {
            println!("Supply a number to square")
        }
    }
}
