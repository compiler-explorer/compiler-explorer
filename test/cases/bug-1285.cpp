#include <bitset>

enum Flags { A = 1, B = 2, C = 3, D = 5,
             E = 8, F = 13, G = 21, H,
             I, J, K, L, M, N, O };

void apply_known_mask(std::bitset<64> &bits) {
    const Flags important_bits[] = { B, D, E, H, K, M, L, O };
    std::remove_reference<decltype(bits)>::type mask{};
    for (const auto& bit : important_bits) {
        mask.set(bit);
    }

    bits &= mask;
}
