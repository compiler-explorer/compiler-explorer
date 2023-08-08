// https://enzyme.mit.edu/getting_started/Examples/#reverse-mode-1

#include <cstdio>

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template <typename return_type, typename... T>
return_type __enzyme_fwddiff(void *, T...);

template <typename return_type, typename... T>
return_type __enzyme_autodiff(void *, T...);

void f(double x, double y, double &output) { output = x * y + 1.0 / y; }

struct double2 {
  double x, y;
};

int main() {
  double x = 3.0;
  double y = 2.0;
  double dx = 1.0;
  double dy = 2.0;
  double z = 0;

  double lambda = 2.0;
#if 0
    double2 mu = __enzyme_autodiff<double2>((void*)f, enzyme_out, x, 
                                                      enzyme_out, y, 
                                                      enzyme_dup, &z, &lambda); 
    printf("z = %f, mu.x = %f, mu.y = %f\n", z, mu.x, mu.y);
#else
  double2 mu = __enzyme_autodiff<double2>((void *)f, enzyme_out, x, enzyme_out,
                                          y, enzyme_dupnoneed, &z, &lambda);
  printf("z = %f, mu.x = %f, mu.y = %f\n", z, mu.x, mu.y);
#endif
}