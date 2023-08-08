// https://enzyme.mit.edu/getting_started/Examples/#forward-mode-1

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

  // these will be overwritten by __enzyme_fwddiff
  double z = 0;
  double dz = 0;
#if 0
    __enzyme_fwddiff<void>((void*)f, enzyme_dup, x, dx, 
                                     enzyme_dup, y, dy, 
                                     enzyme_dup, &z, &dz);
    printf("f(x,y) = %f, df = %f\n", z, dz);
#else
  __enzyme_fwddiff<void>((void *)f, enzyme_dup, x, dx, enzyme_dup, y, dy,
                         enzyme_dupnoneed, &z, &dz);
  printf("f(x,y) = %f, df = %f\n", z, dz);
#endif
}