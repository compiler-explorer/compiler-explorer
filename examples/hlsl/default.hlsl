// The entry point and target profile are needed to compile this example:
// DXC:     -T ps_6_6 -E PSMain
// AMD RGA: -s dx12 -c gfx1201 --ps-model ps_6_6 --ps-entry PSMain

struct PSInput
{
    float4 position : SV_Position;
    float4 color    : COLOR0;
};

float4 PSMain(PSInput input) : SV_Target0
{
    return input.color * input.color;
}
