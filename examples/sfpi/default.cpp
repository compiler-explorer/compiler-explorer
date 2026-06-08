#include <cstdint>

extern volatile std::uint32_t __instrn_buffer[];

namespace ckernel {
constexpr inline volatile std::uint32_t *[[gnu::rvtt_reg_ptr]] instrn_buffer = ::__instrn_buffer;
}

#include "sfpi.h"

volatile std::uint32_t __instrn_buffer[64];

using namespace sfpi;

extern "C" void sfpi_demo() {
    vFloat a = dst_reg[0];
    vFloat b = dst_reg[1];
    dst_reg[2] = a * b + 1.0f;
    dst_reg += 1;
}
