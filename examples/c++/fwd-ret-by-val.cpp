// https://enzyme.mit.edu/getting_started/Examples/#forward-mode

#include <cstdio>

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template <typename return_type, typename... T>
return_type __enzyme_fwddiff(void *, T...);

template <typename return_type, typename... T>
return_type __enzyme_autodiff(void *, T...);

double f(double x) { return x * x; }

int main() {
  double x = 5.0;
  double dx = 1.0;
  double df_dx = __enzyme_fwddiff<double>((void *)f, enzyme_dup, x, dx);
  printf("f(x) = %f, f'(x) = %f", f(x), df_dx);
}