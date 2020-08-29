#include <type_traits>
#include <thread>

struct safetls
{
    int val;
    int x;
    int y;
    int value() {return val - x + y;}
};

thread_local safetls g{0x98765};
thread_local int h = 0x12345;

inline int func()
{
    return g.value();
}

inline int func2()
{
    return h;
}

int main()
{
    return func() + func2();
}