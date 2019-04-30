#include <typeinfo>

int foo() { return 42; }
int bar() { return typeid(int).name()[0]; }
class baz_t {};
int baz() { return typeid(baz_t).name()[0]; }
