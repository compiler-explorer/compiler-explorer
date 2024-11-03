#![feature(autodiff)]
use std::autodiff::autodiff;

// See https://enzyme.mit.edu/index.fcgi/rust/usage/usage.html
// for more details
#[autodiff(dsquare, Reverse, Active, Active)]
pub fn square(num: f32) -> f32 {
    num * num
}

fn main(){
    for i in 0..5 {
        // This also returns the derivative of square, 2 * x
        dbg!(dsquare(i as f32, 1.0));
    }
}
