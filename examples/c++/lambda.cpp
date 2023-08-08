// https://enzyme.mit.edu/getting_started/Examples/#functors-and-lambda-functions

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
  double operator()(double y) const { return x * y + 1.0 / y; }
  double x;
};

template <typename T, typename... arg_types>
auto wrapper(const T &f, arg_types &&...args) {
  return f(args...);
}

// C++20 or later
template <auto obj, typename... arg_types> auto wrapper(arg_types &&...args) {
  return obj(args...);
}

int main() {
  MyObject obj{1.0};
  auto f = [](double x, double y) { return x * y + 1.0 / y; };

  double x = 1.0;
  double dx = 3.1;
  double y = 2.0;
  double dy = 1.0;

  {
    double dfdy =
        __enzyme_fwddiff<double>((void *)(wrapper<MyObject, double>),
                                 enzyme_const, &obj, enzyme_dup, &y, &dy);
    printf("dfdy = %f\n", dfdy);
  }

  {
    double dfdx = __enzyme_fwddiff<double>((void *)+f, enzyme_dup, x, dx,
                                           enzyme_const, y);
    printf("dfdx = %f\n", dfdx);
  }

  {
    double dfdy =
        __enzyme_fwddiff<double>((void *)(wrapper<f, double, double>),
                                 enzyme_const, &x, enzyme_dup, &y, &dy);
    printf("dfdy = %f\n", dfdy);
  }

  {
    double dfdy = __enzyme_fwddiff<double>(
        (void *)(wrapper<decltype(f), double, double>), enzyme_const,
        (void *)&f, enzyme_const, &x, enzyme_dup, &y, &dy);
    printf("dfdy = %f\n", dfdy);
  }
}