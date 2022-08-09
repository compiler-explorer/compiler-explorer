// The entry point and target profile are needed to compile this example:
// -T ps_6_6 -E PSMain

struct PSInput
{
    float4 position : SV_Position;
    float4 color    : COLOR0;
};

float4 PSMain(PSInput input) : SV_Target0
{
    return input.color * input.color;
}
