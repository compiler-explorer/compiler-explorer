#include <Groundfloor/Molecules/String.h>

int square(int num) {
    Groundfloor::String s("Hello, world!");
    s.append_ansi("\n");

    printf(s.getValue());

    return num * num;
}

int main(int argc) {
    return square(argc);
}


