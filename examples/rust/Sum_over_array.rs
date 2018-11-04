pub fn sum_array_loop(input: &[i32]) -> i32 {
  let mut sum: i32 = 0;

  for i in 0..input.len() {
    sum += input[i];
  }

  sum
}

pub fn sum_array_iterator(input: &[i32]) -> i32 {
  input.iter().sum()
}
