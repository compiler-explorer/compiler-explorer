// compile flags:
// [amd64] cl /FA /EHsc /c vc-main-opt-out-example.cpp /FAvc-main-opt-out.asm
// then, replace the filename for <array> with std\array
#include <array>

template <int... vars>
constexpr auto make_array() {
    return std::array<int, sizeof...(vars)>{vars...};
}

int main() {
    constexpr auto a = make_array<1, 3, 4, 2, 3, 5, 6, 4, 8, 9, 2>();

    return a[2];
}