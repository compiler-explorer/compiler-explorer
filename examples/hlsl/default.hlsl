// The entry point and target profile are needed to compile this example:
// -T ps_6_6 -E PSMain
Texture2D<float4> color : register(t0, space0);

struct Constants
{
    uint sampler_index;
};
ConstantBuffer<Constants> constants : register(b0, space0);

struct PSInput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

SamplerState color_sampler : register(s0, space0);

float4 PSMain(PSInput input) : SV_Target0
{
    return color.Sample(color_sampler, input.uv);
}
