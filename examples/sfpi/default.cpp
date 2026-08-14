// Required for standalone CE SFPI examples; it provides the firmware-side compat definitions.
#include "ce_sfpi_compat.h"

#include "sfpi.h"

using namespace sfpi;

extern "C" void sfpi_demo() {
    vFloat a = dst_reg[0];
    vFloat b = dst_reg[1];
    dst_reg[2] = a * b + 1.0f;
    dst_reg += 1;
}
