use std::io;
use std::io::prelude::*;
use regex::Regex;
use regex::Captures;
use rustc_demangle::demangle;

extern crate rustc_demangle;
extern crate regex;

fn main() {
    let stdin = io::stdin();
    let token = Regex::new(r"[_a-zA-Z$][_a-zA-Z$0-9]*").unwrap();

    for line in stdin.lock().lines() {
        println!("{}",
                 line.unwrap()
                     .split_whitespace()
                     .map(|x| demangle(x).to_string())
                     .collect::<Vec<String>>()
                     .join(" "));
//        println!("{}", token.replace(line.unwrap(), |caps: &Captures| {
//            demangle(caps.at(0).unwrap()).to_string()
//        }));
    }
}