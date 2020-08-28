// compile flags:
// [arm] cl /O2 /FA /c arm-moose-example.cpp /Faarm-moose.asm

extern "C" {
    static int thing(int a) {
        return a * a;
    }

    int moose(int x, int y) {
        int foo = 1;

        for (int i = 0; i < y; ++i) {
            foo += thing(x);
        }
        return foo;
    }
}
