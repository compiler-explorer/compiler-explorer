use std::env;

const fn square(num: i32) -> i32 {
    num * num
}

pub fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let r: i32 = args[1].parse::<i32>().unwrap();
        let sq = square(r);
        println!("{}", sq);
    } else {
        println!("Supply a number to square")
    }
}
