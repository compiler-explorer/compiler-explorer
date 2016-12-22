use std::io;
use std::io::prelude::*;
use rustc_demangle::demangle;

extern crate rustc_demangle;

fn main() {
    let stdin = io::stdin();

    for line in stdin.lock().lines() {
        let a = line.unwrap();
        println!("{}", demangle(&a));
    }
    /*
                    let var: Vec<String> = input.split(' ')
                    .into_iter()
                    .map(|x| demangle(x).to_string())
                    .collect();
                println!("{}", var.join(" "));
*/
}