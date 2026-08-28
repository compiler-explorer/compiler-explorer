// Required for standalone CE SFPI examples; it provides the firmware-side compat definitions.
#include "ce_sfpi_compat.h"

#include "sfpi.h"

using namespace sfpi;

extern "C" void sfpi_control_flow_demo(bool take_abs) {
    vFloat a = dst_reg[0] + vFloat(2.0F);

    vFloat b = vFloat(dst_reg[1]);
    dst_reg[3] = a * -b + vConstFloatPrgm0 + 0.5F;

    dst_reg[4] = a * dst_reg[1] + 1.2F;
    dst_reg[4] = a * 1.5F + 1.2F;

    vFloat tmp = sFloat16a(2);
    dst_reg[5] = a * tmp + 1.2F;

    v_if ((a >= 4.0F && a < 8.0F) || (a >= 12.0F && a < 16.0F)) {
        vInt b = exexp(a, sfpi::ExponentMode::NoDebias);
        b &= 0xAA;
        v_if (b >= 130) {
            dst_reg[6] = setexp(a, 127);
        }
        v_endif;
    } v_elseif (a == sFloat16a(3)) {
        if (take_abs) {
            dst_reg[7] = abs(a);
        } else {
            dst_reg[7] = a;
        }
    } v_else {
        dst_reg[8] = -setexp(a, 126);
    }
    v_endif;
}
