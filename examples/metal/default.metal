// The standard version is needed to compile this example:
// -std=metal3.1

#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

fragment float4 fragmentShader(VertexOut in [[stage_in]]) {
    return in.color * in.color;
}
