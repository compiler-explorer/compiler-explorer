#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct main0_out
{
    float4 outFragColor [[color(0)]];
};

struct main0_in
{
    float3 inColor [[user(locn0)]];
};

fragment main0_out main0(main0_in in [[stage_in]])
{
    main0_out out = {};
    out.outFragColor = float4(in.inColor, 1.0);
    return out;
}
