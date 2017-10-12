pub fn sum_array(x: &[i32]) -> i32 {
  x.iter().fold(0, |sum, next| sum + *next)
}
