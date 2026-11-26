# main.cpp
#include <cstdio>
int square(int x);
int main() { std::printf("%d\n", square(7)); }
# sq.cpp (static library target sq)
int square(int x) { return x * x; }
