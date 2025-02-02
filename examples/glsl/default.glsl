// This will be consumed as a .glsl file and needs the stage and target profile. Example options:
// -S comp --target-env vulkan1.1
#version 450
layout(set = 0, binding = 0, rgba8) readonly uniform image2D myImage;
layout(set = 0, binding = 1, std430) buffer SSBO {
    ivec2 coords;
    vec4 data;
};

void main() {
    data = imageLoad(myImage, coords);
}
