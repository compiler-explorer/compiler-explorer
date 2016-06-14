// Compile with -O3 -march=native to see autovectorization
// assumes input is aligned on 64-byte boundary and that
// length is a multiple of 64.
int testFunction(int* input, int length) {
  // Alignment hints supported on GCC 4.7+ and any compiler
  // supporting the appropriate builtin (clang 3.6+).
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif
#if __GNUC__ > 4 \
    || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7) \
    || __has_builtin(__builtin_assume_aligned)
  input = static_cast<int*>(__builtin_assume_aligned(input, 64));
#endif
#if _MSC_VER
  __assume((length & 63) == 0);
#else
  if (length & 63) __builtin_unreachable();
#endif
  int sum = 0;
  for (int i = 0; i < length; ++i) {
    sum += input[i];
  }
  return sum;
}
