// https://enzyme.mit.edu/getting_started/Examples/#reverse-mode

#include <cstdio>

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template <typename return_type, typename... T>
return_type __enzyme_fwddiff(void *, T...);

template <typename return_type, typename... T>
return_type __enzyme_autodiff(void *, T...);

double f(double x, double y) { return x * y + 1.0 / y; }

struct double2 {
  double x, y;
};

int main() {
  double x = 3.0;
  double y = 2.0;
  double2 mu =
      __enzyme_autodiff<double2>((void *)f, enzyme_out, x, enzyme_out, y);
  printf("mu.x = %f, mu.y = %f\n", mu.x, mu.y);
}