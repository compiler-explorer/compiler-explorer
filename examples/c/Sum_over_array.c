int testFunction(int* input, int length) {
  int i, sum = 0;

  for (i = 0; i < length; ++i) {
    sum += input[i];
  }
  return sum;
}
