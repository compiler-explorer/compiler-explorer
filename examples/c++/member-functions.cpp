// https://enzyme.mit.edu/getting_started/Examples/#member-functions

#include <cstdio>

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template <typename return_type, typename... T>
return_type __enzyme_fwddiff(void *, T...);

template <typename return_type, typename... T>
return_type __enzyme_autodiff(void *, T...);

struct MyObject {
  double f(double y) { return x * y + 1.0 / y; }
  double x;
};

template <typename T, typename... arg_types>
auto wrapper(T obj, arg_types &&...args) {
  return obj.f(args...);
}

int main() {
  MyObject obj{3.0};

  double y = 2.0;
  double dy = 2.0;

  double dfdy =
      __enzyme_fwddiff<double>((void *)wrapper<MyObject, double>, enzyme_const,
                               obj, enzyme_dup, &y, &dy);
  printf("dfdy = %f\n", dfdy);
}