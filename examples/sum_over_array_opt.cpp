// Compile with -O3 to see autovectorization
int testFunction(int* input, int length) {
#if __GNUC_MINOR__ >= 7
  // gcc 4.7 allows us to tell it about alignments.
  input = static_cast<int*>(__builtin_assume_aligned(input, 16));
#endif
  int sum = 0;
  for (int i = 0; i < length; ++i) {
    sum += input[i];
  }
  return sum;
}
