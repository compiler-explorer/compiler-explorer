#[macro_use] extern crate lazy_static;
extern crate rustc_demangle;
extern crate regex;

use std::io;
use std::io::prelude::*;
use regex::Regex;
use regex::Captures;
use rustc_demangle::demangle;

fn demangle_line(line: &str) -> String {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"[_a-zA-Z$][_a-zA-Z$0-9.]*").unwrap();
    }

    RE.replace_all(line, |caps: &Captures| {
        format!("{:#}", demangle(caps.get(0).unwrap().as_str()))
    }).to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn passes_text() {
        assert_eq!(
        ::demangle_line("mo fo\tboom      hello  "),
        "mo fo\tboom      hello  ");
    }

    #[test]
    fn demangles() {
        assert_eq!(
        ::demangle_line("_ZN7example4main17h0db00b8b32acffd5E:"),
        "example::main:");
    }

    #[test]
    fn handles_mid_demangling() {
        assert_eq!(
        ::demangle_line("        lea     rax, [rip + _ZN55_$LT$$RF$$u27$a$u20$T$u20$as$u20$core..fmt..Display$GT$3fmt17h510ed05e72307174E]"),
        "        lea     rax, [rip + <&\'a T as core::fmt::Display>::fmt]");
    }

    #[test]
    fn handles_call_plt() {
        assert_eq!(
        ::demangle_line("        call    _ZN3std2io5stdio6_print17he48522be5b0a80d9E@PLT"),
        "        call    std::io::stdio::_print@PLT");
    }
}

fn main() {
    let stdin = io::stdin();

    for line in stdin.lock().lines() {
        println!("{}", demangle_line(&line.unwrap()));
    }
}
