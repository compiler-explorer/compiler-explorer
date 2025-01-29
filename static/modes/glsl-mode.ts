// Copyright (c) 2024, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import $ from 'jquery';

import * as monaco from 'monaco-editor';
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore  "Could not find a declaration file"
import * as cpp from 'monaco-editor/esm/vs/basic-languages/cpp/cpp';

function definition(): monaco.languages.IMonarchLanguage {
    const glsl = $.extend(true, {}, cpp.language);

    function addKeywords(keywords: string[]) {
        for (let i = 0; i < keywords.length; ++i) {
            glsl.keywords.push(keywords[i]);
        }
    }

    function matrixTypes(basename: string) {
        return [
            `${basename}2`,
            `${basename}3`,
            `${basename}4`,
            `${basename}2x2`,
            `${basename}2x3`,
            `${basename}2x4`,
            `${basename}3x2`,
            `${basename}3x3`,
            `${basename}3x4`,
            `${basename}4x2`,
            `${basename}4x3`,
            `${basename}4x4`,
        ];
    }

    function vectorTypes(basename: string) {
        return [`${basename}2`, `${basename}3`, `${basename}4`];
    }

    addKeywords(matrixTypes('mat'));
    addKeywords(matrixTypes('dmat'));
    addKeywords(vectorTypes('vec'));
    addKeywords(vectorTypes('uvec'));
    addKeywords(vectorTypes('ivec'));
    addKeywords(vectorTypes('bvec'));
    addKeywords(vectorTypes('dvec'));
    // GL_EXT_shader_explicit_arithmetic_types_float64
    addKeywords(vectorTypes('f64vec'));
    addKeywords(matrixTypes('f64mat'));
    // GL_EXT_shader_explicit_arithmetic_types_float32
    addKeywords(vectorTypes('f32vec'));
    addKeywords(matrixTypes('f32mat'));
    // GL_EXT_shader_explicit_arithmetic_types_float16
    addKeywords(vectorTypes('f16vec'));
    addKeywords(matrixTypes('f16mat'));
    // GL_EXT_shader_explicit_arithmetic_types_int64
    addKeywords(vectorTypes('i64vec'));
    addKeywords(vectorTypes('u64vec'));
    // GL_EXT_shader_explicit_arithmetic_types_int32
    addKeywords(vectorTypes('i32vec'));
    addKeywords(vectorTypes('u32vec'));
    // GL_EXT_shader_explicit_arithmetic_types_int16
    addKeywords(vectorTypes('i16vec'));
    addKeywords(vectorTypes('u16vec'));
    // GL_EXT_shader_explicit_arithmetic_types_int8
    addKeywords(vectorTypes('i8vec'));
    addKeywords(vectorTypes('u8vec'));

    function samplerTypes(basename: string) {
        return [`${basename}`, `${basename}Shadow`, `${basename}Array`, `${basename}ArrayShadow`];
    }

    addKeywords(samplerTypes('sampler1D'));
    addKeywords(samplerTypes('isampler1D'));
    addKeywords(samplerTypes('sampler2D'));
    addKeywords(samplerTypes('samplerCube'));

    function imageTypes(basename: string) {
        return [
            `${basename}1D`,
            `${basename}1DArray`,
            `${basename}2D`,
            `${basename}2DArray`,
            `${basename}2DRect`,
            `${basename}2DMS`,
            `${basename}2DMSArray`,
            `${basename}3D`,
            `${basename}Cube`,
            `${basename}CubeArray`,
            `${basename}Buffer`,
        ];
    }
    addKeywords(imageTypes('image'));
    addKeywords(imageTypes('iimage'));
    addKeywords(imageTypes('uimage'));
    addKeywords(imageTypes('texture'));
    addKeywords(imageTypes('itexture'));
    addKeywords(imageTypes('utexture'));
    // GLSL_EXT_shader_image_int64
    addKeywords(imageTypes('u64image'));
    addKeywords(imageTypes('i64image'));

    addKeywords([
        'attribute',
        'uniform',
        'buffer',
        'sampler',
        'varying',
        'lowp',
        'mediump',
        'highp',
        'precision',
        'invariant',
        'discard',
        'subroutine',
        'layout',
        'location',
        'flat',
        'centroid',
        'texture',
        'sample',
        'patch',
        'points',
        'lines',
        'triangles',
        'max_vertices',
        'max_primitives',
        'local_size_x',
        'local_size_y',
        'local_size_z',
        'local_size_x_id',
        'local_size_y_id',
        'local_size_z_id',
        'precise',
        'noperspective',
        'writeonly',
        'readonly',
        'in',
        'out',
        'inout',

        // scalar types
        'atomic_uint',
        'float16_t',
        'float32_t',
        'float64_t',
        'int64_t',
        'uint64_t',
        'int32_t',
        'uint32_t',
        'int16_t',
        'uint16_t',
        'int8_t',
        'uint8_t',

        // GLSL.std.450.pdf
        'radians',
        'degrees',
        'sin',
        'cos',
        'tan',
        'asin',
        'acos',
        'atan',
        'sinh',
        'cosh',
        'tanh',
        'asinh',
        'acosh',
        'atanh',
        'pow',
        'exp',
        'log',
        'exp2',
        'log2',
        'sqrt',
        'inversesqrt',
        'abs',
        'sign',
        'floor',
        'trunc',
        'round',
        'roundEven',
        'ceil',
        'fract',
        'mod',
        'modf',
        'min',
        'max',
        'clamp',
        'mix',
        'step',
        'smoothstep',
        'isnan',
        'isinf',
        'floatBitsToInt',
        'floatBitsToUint',
        'intBitsToFloat',
        'uintBitsToFloat',
        'fma',
        'frexp',
        'ldexp',
        'packUnorm2x16',
        'packSnorm2x16',
        'packUnorm4x8',
        'packSnorm4x8',
        'unpackUnorm2x16',
        'unpackSnorm2x16',
        'unpackUnorm4x8',
        'unpackSnorm4x8',
        'packHalf2x16',
        'unpackHalf2x16',
        'packDouble2x32',
        'unpackDouble2x32',
        'length',
        'distance',
        'dot',
        'cross',
        'normalize',
        'faceforward',
        'reflect',
        'refract',
        'matrixCompMult',
        'outerProduct',
        'transpose',
        'determinant',
        'inverse',
        'lessThan',
        'lessThanEqual',
        'greaterThan',
        'greaterThanEqual',
        'equal',
        'notEqual',
        'any',
        'all',
        'not',
        'uaddCarry',
        'usubBorrow',
        'umulExtended',
        'bitfieldExtract',
        'bitfieldInsert',
        'bitfieldReverse',
        'bitCount',
        'findLSB',
        'findMSB',
        'dFdx',
        'dFdxFine',
        'dFdxCoarse',
        'dFdy',
        'dFdyFine',
        'dFdyCoarse',
        'fwidth',
        'fwidthFine',
        'fwidthCoarse',
        'anyInvocation',
        'allInvocations',
        'allInvocationsEqual',

        // misc samplers
        'isamplerCube',
        'isamplerCubeArray',
        'usamplerCube',
        'usamplerCubeArray',
        'isampler3D',
        'usampler3D',
        'sampler3D',
        'sampler2DRect',
        'sampler2DRectShadow',
        'isampler2DRect',
        'usampler2DRect',
        'sampler2DMS',
        'isampler2DMS',
        'usampler2DMS',
        'sampler2DMSArray',
        'isampler2DMSArray',
        'usampler2DMSArray',
        'samplerBuffer',
        'isamplerBuffer',
        'usamplerBuffer',
        'samplerShadow',

        // misc texture
        'textureProjLod',
        'textureProj',
        'textureLod',
        'textureOffset',
        'texelFetch',
        'texelFetchOffset',
        'textureGather',
        'textureGatherOffset',
        'textureProjOffset',
        'textureLodOffset',
        'textureGrad',
        'textureGradOffset',
        'textureProjGrad',
        'textureProjGradOffset',
        'textureSize',
        'textureQueryLod',
        'textureQueryLevels',
        'texture1DProj',
        'texture1DLod',
        'texture1DProjLod',
        'texture2DProj',
        'texture2DLod',
        'texture2DProjLod',
        'texture3DProj',
        'texture3DLod',
        'texture3DProjLod',
        'textureCube',
        'textureCubeLod',
        'shadow1D',
        'shadow1DProj',
        'shadow1DLod',
        'shadow1DProjLod',
        'shadow2D',
        'shadow2DProj',
        'shadow2DLod',
        'shadow2DProjLod',

        // subpasses
        'subpassInput',
        'isubpassInput',
        'usubpassInput',
        'subpassInputMS',
        'isubpassInputMS',
        'usubpassInputMS',
        'subpassLoad',

        // atomics
        'atomicCounterIncrement',
        'atomicCounterDecrement',
        'atomicCounter',
        'atomicCounterAdd',
        'atomicCounterSubtract',
        'atomicCounterMin',
        'atomicCounterMax',
        'atomicCounterAnd',
        'atomicCounterOr',
        'atomicCounterXor',
        'atomicCounterExchange',
        'atomicCounterCompSwap',
        'atomicAdd',
        'atomicMin',
        'atomicMax',
        'atomicAnd',
        'atomicOr',
        'atomicXor',
        'atomicExchange',
        'atomicCompSwap',
        'imageAtomicAdd',
        'imageAtomicMin',
        'imageAtomicMax',
        'imageAtomicAnd',
        'imageAtomicOr',
        'imageAtomicXor',
        'imageAtomicExchange',
        'imageAtomicCompSwap',

        // misc images
        'imageSize',
        'imageSamples',
        'imageLoad',
        'imageStore',

        // Geometry
        'EmitStreamVertex',
        'EndStreamPrimitive',
        'EmitVertex',
        'EndPrimitive',

        'interpolateAtCentroid',
        'interpolateAtSample',
        'interpolateAtOffset',

        'noise1',
        'noise2',
        'noise3',
        'noise4',

        // barriers
        'barrier',
        'memoryBarrier',
        'memoryBarrierAtomicCounter',
        'memoryBarrierBuffer',
        'memoryBarrierShared',
        'memoryBarrierImage',
        'groupMemoryBarrier',
        'controlBarrier',
        // GL_KHR_memory_scope_semantics
        'coherent',
        'devicecoherent',
        'queuefamilycoherent',
        'workgroupcoherent',
        'subgroupcoherent',
        'nonprivate',
        'volatile',

        // built-ins Everyone
        'gl_in',
        'gl_out',

        // built-ins Vertex
        'gl_VertexID',
        'gl_InstanceID',
        'gl_VertexIndex',
        'gl_InstanceIndex',
        'gl_DrawID',
        'gl_BaseVertex',
        'gl_BaseInstance',
        'gl_PerVertex',
        'gl_Position',
        'gl_PointSize',
        'gl_ClipDistance',
        'gl_CullDistance',

        // built-ins Tessellation / Geometry
        'gl_TessLevelOuter',
        'gl_TessLevelInner',
        'gl_PatchVerticesIn',
        'gl_InvocationID',
        'gl_TessCoord',
        'gl_PrimitiveIDIn',

        // built-ins Frag
        'gl_FragCoord',
        'gl_FrontFacing',
        'gl_PointCoord',
        'gl_PrimitiveID',
        'gl_SampleID',
        'gl_SamplePosition',
        'gl_SampleMaskIn',
        'gl_Layer',
        'gl_ViewportIndex',
        'gl_HelperInvocation',
        'gl_FragDepth',
        'gl_SampleMask',

        // built-ins Compute
        'gl_NumWorkGroups',
        'gl_WorkGroupSize',
        'gl_WorkGroupID',
        'gl_LocalInvocationID',
        'gl_GlobalInvocationID',
        'gl_LocalInvocationIndex',

        // limits
        'gl_MaxVertexAttribs',
        'gl_MaxVertexUniformVectors',
        'gl_MaxVertexUniformComponents',
        'gl_MaxVertexOutputComponents',
        'gl_MaxVaryingComponents',
        'gl_MaxVaryingVectors',
        'gl_MaxVertexTextureImageUnits',
        'gl_MaxVertexImageUniforms',
        'gl_MaxVertexAtomicCounters',
        'gl_MaxVertexAtomicCounterBuffers',
        'gl_MaxTessPatchComponents',
        'gl_MaxPatchVertices',
        'gl_MaxTessGenLevel',
        'gl_MaxTessControlInputComponents',
        'gl_MaxTessControlOutputComponents',
        'gl_MaxTessControlTextureImageUnits',
        'gl_MaxTessControlUniformComponents',
        'gl_MaxTessControlTotalOutputComponents',
        'gl_MaxTessControlImageUniforms',
        'gl_MaxTessControlAtomicCounters',
        'gl_MaxTessControlAtomicCounterBuffers',
        'gl_MaxTessEvaluationInputComponents',
        'gl_MaxTessEvaluationOutputComponents',
        'gl_MaxTessEvaluationTextureImageUnits',
        'gl_MaxTessEvaluationUniformComponents',
        'gl_MaxTessEvaluationImageUniforms',
        'gl_MaxTessEvaluationAtomicCounters',
        'gl_MaxTessEvaluationAtomicCounterBuffers',
        'gl_MaxGeometryInputComponents',
        'gl_MaxGeometryOutputComponents',
        'gl_MaxGeometryImageUniforms',
        'gl_MaxGeometryTextureImageUnits',
        'gl_MaxGeometryOutputVertices',
        'gl_MaxGeometryTotalOutputComponents',
        'gl_MaxGeometryUniformComponents',
        'gl_MaxGeometryVaryingComponents',
        'gl_MaxGeometryAtomicCounters',
        'gl_MaxGeometryAtomicCounterBuffers',
        'gl_MaxFragmentImageUniforms',
        'gl_MaxFragmentInputComponents',
        'gl_MaxFragmentUniformVectors',
        'gl_MaxFragmentUniformComponents',
        'gl_MaxFragmentAtomicCounters',
        'gl_MaxFragmentAtomicCounterBuffers',
        'gl_MaxDrawBuffers',
        'gl_MaxTextureImageUnits',
        'gl_MinProgramTexelOffset',
        'gl_MaxProgramTexelOffset',
        'gl_MaxImageUnits',
        'gl_MaxSamples',
        'gl_MaxImageSamples',
        'gl_MaxClipDistances',
        'gl_MaxCullDistances',
        'gl_MaxViewports',
        'gl_MaxComputeImageUniforms',
        'gl_MaxComputeWorkGroupCount',
        'gl_MaxComputeWorkGroupSize',
        'gl_MaxComputeUniformComponents',
        'gl_MaxComputeTextureImageUnits',
        'gl_MaxComputeAtomicCounters',
        'gl_MaxComputeAtomicCounterBuffers',
        'gl_MaxCombinedTextureImageUnits',
        'gl_MaxCombinedImageUniforms',
        'gl_MaxCombinedImageUnitsAndFragmentOutputs',
        'gl_MaxCombinedShaderOutputResources',
        'gl_MaxCombinedAtomicCounters',
        'gl_MaxCombinedAtomicCounterBuffers',
        'gl_MaxCombinedClipAndCullDistances',
        'gl_MaxAtomicCounterBindings',
        'gl_MaxAtomicCounterBufferSize',
        'gl_MaxTransformFeedbackBuffers',
        'gl_MaxTransformFeedbackInterleavedComponents',
        'gl_MaxInputAttachments',

        // GLSL_EXT_buffer_reference
        'buffer_reference',
        'buffer_reference_align',
        // GLSL_EXT_debug_printf
        'debugPrintfEXT',
        // GLSL_EXT_demote_to_helper_invocation
        'demote',
        // GLSL_EXT_fragment_invocation_density
        'gl_FragSizeEXT',
        'gl_FragInvocationCountEXT',
        // GLSL_EXT_fragment_shader_barycentric
        'pervertexEXT',
        // GLSL_EXT_fragment_shading_rate
        'gl_ShadingRateEXT',
        'gl_PrimitiveShadingRateEXT',
        'gl_ShadingRateFlag2VerticalPixelsEXT',
        'gl_ShadingRateFlag4VerticalPixelsEXT',
        'gl_ShadingRateFlag2HorizontalPixelsEXT',
        'gl_ShadingRateFlag4HorizontalPixelsEXT',
        // GLSL_EXT_mesh_shader
        'taskPayloadSharedEXT',
        'perprimitiveEXT',
        'EmitMeshTasksEXT',
        'SetMeshOutputsEXT',
        'gl_PrimitivePointIndicesEXT',
        'gl_PrimitiveLineIndicesEXT',
        'gl_PrimitiveTriangleIndicesEXT',
        'gl_CullPrimitiveEXT',
        'gl_MeshPerVertexEXT',
        'gl_MeshPerPrimitiveEXT',
        // GLSL_EXT_opacity_micromap
        'gl_RayFlagsForceOpacityMicromap2StateEXT',
        // GLSL_EXT_ray_cull_mask
        'gl_CullMaskEXT',
        // GLSL_EXT_ray_flags_primitive_culling
        'gl_RayFlagsSkipTrianglesEXT',
        'gl_RayFlagsSkipAABBEXT',
        // GLSL_EXT_ray_query
        'accelerationStructureEXT',
        'RayQueryCommittedIntersectionKHR',
        'gl_RayFlagsNoneEXT',
        'gl_RayFlagsOpaqueEXT',
        'gl_RayFlagsNoOpaqueEXT',
        'gl_RayFlagsTerminateOnFirstHitEXT',
        'gl_RayFlagsSkipClosestHitShaderEXT',
        'gl_RayFlagsCullBackFacingTrianglesEXT',
        'gl_RayFlagsCullFrontFacingTrianglesEXT',
        'gl_RayFlagsCullOpaqueEXT',
        'gl_RayFlagsCullNoOpaqueEXT',
        'gl_RayQueryCommittedIntersectionNoneEXT',
        'gl_RayQueryCommittedIntersectionTriangleEXT',
        'gl_RayQueryCommittedIntersectionGeneratedEXT',
        'gl_RayQueryCandidateIntersectionTriangleEXT',
        'gl_RayQueryCandidateIntersectionAABBEXT',
        'rayQueryEXT',
        'rayQueryInitializeEXT',
        'rayQueryProceedEXT',
        'rayQueryTerminateEXT',
        'rayQueryGenerateIntersectionEXT',
        'rayQueryConfirmIntersectionEXT',
        'rayQueryGetIntersectionTypeEXT',
        'rayQueryGetRayTMinEXT',
        'rayQueryGetRayFlagsEXT',
        'rayQueryGetWorldRayOriginEXT',
        'rayQueryGetWorldRayDirectionEXT',
        'rayQueryGetIntersectionTEXT',
        'rayQueryGetIntersectionInstanceCustomIndexEXT',
        'rayQueryGetIntersectionInstanceIdEXT',
        'rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT',
        'rayQueryGetIntersectionGeometryIndexEXT',
        'rayQueryGetIntersectionPrimitiveIndexEXT',
        'rayQueryGetIntersectionBarycentricsEXT',
        'rayQueryGetIntersectionFrontFaceEXT',
        'rayQueryGetIntersectionCandidateAABBOpaqueEXT',
        'rayQueryGetIntersectionObjectRayDirectionEXT',
        'rayQueryGetIntersectionObjectRayOriginEXT',
        'rayQueryGetIntersectionObjectToWorldEXT',
        'rayQueryGetIntersectionWorldToObjectEXT',
        // GLSL_EXT_ray_tracing
        'rayPayloadEXT',
        'rayPayloadInEXT',
        'hitAttributeEXT',
        'callableDataEXT',
        'callableDataInEXT',
        'shaderRecordEXT',
        'gl_LaunchIDEXT',
        'gl_LaunchSizeEXT',
        'gl_InstanceCustomIndexEXT',
        'gl_GeometryIndexEXT',
        'gl_WorldRayOriginEXT',
        'gl_WorldRayDirectionEXT',
        'gl_ObjectRayOriginEXT',
        'gl_ObjectRayDirectionEXT',
        'gl_RayTminEXT',
        'gl_RayTmaxEXT',
        'gl_IncomingRayFlagsEXT',
        'gl_HitTEXT',
        'gl_HitKindEXT',
        'gl_ObjectToWorldEXT',
        'gl_WorldToObjectEXT',
        'gl_WorldToObject3x4EXT',
        'gl_ObjectToWorld3x4EXT',
        'gl_HitKindFrontFacingTriangleEXT',
        'gl_HitKindBackFacingTriangleEXT',
        'traceRayEXT',
        'reportIntersectionEXT',
        'ignoreIntersectionEXT',
        'terminateRayEXT',
        'executeCallableEXT',
        'shadercallcoherent',
        // GLSL_EXT_ray_tracing_position_fetch
        'gl_HitTriangleVertexPositionsEXT',
        'rayQueryGetIntersectionTriangleVertexPositionsEXT',
        // GLSL_EXT_shader_subgroup_extended_types
        'subgroupAllEqual',
        'subgroupBroadcast',
        'subgroupBroadcastFirst',
        'subgroupShuffle',
        'subgroupShuffleXor',
        'subgroupShuffleUp',
        'subgroupShuffleDown',
        'subgroupAdd',
        'subgroupMul',
        'subgroupMin',
        'subgroupMax',
        'subgroupAnd',
        'subgroupOr',
        'subgroupXor',
        'subgroupInclusiveAdd',
        'subgroupInclusiveMul',
        'subgroupInclusiveMin',
        'subgroupInclusiveMax',
        'subgroupInclusiveAnd',
        'subgroupInclusiveOr',
        'subgroupInclusiveXor',
        'subgroupExclusiveAdd',
        'subgroupExclusiveMul',
        'subgroupExclusiveMin',
        'subgroupExclusiveMax',
        'subgroupExclusiveAnd',
        'subgroupExclusiveOr',
        'subgroupExclusiveXor',
        'subgroupClusteredAdd',
        'subgroupClusteredMul',
        'subgroupClusteredMin',
        'subgroupClusteredMax',
        'subgroupClusteredAnd',
        'subgroupClusteredOr',
        'subgroupClusteredXor',
        'subgroupQuadBroadcast',
        'subgroupQuadSwapHorizontal',
        'subgroupQuadSwapVertical',
        'subgroupQuadSwapDiagonal',
        // GLSL_EXT_shader_subgroup_extended_types
        'tileImageEXT',
        'attachmentEXT',
        'iattachmentEXT',
        'uattachmentEXT',
        // GLSL_EXT_spirv_intrinsics
        'spirv_instruction',
        'spirv_execution_mode',
        'spirv_execution_mode_id',
        'spirv_decorate',
        'spirv_decorate_id',
        'spirv_decorate_string',
        'spirv_type',
        'spirv_storage_class',
        'spirv_by_reference',
        'spirv_literal',
        // GL_EXT_nonuniform_qualifier
        'nonuniformEXT',
        // GL_EXT_subgroupuniform_qualifier
        'subgroupuniformEXT',
        // GL_EXT_terminate_invocation
        'terminateInvocation',
        // GLSL_KHR_cooperative_matrix
        'coopmat',
        'coopmatLoad',
        'coopmatStore',
        'coopmatMulAdd',
        // GL_KHR_shader_subgroup
        'gl_NumSubgroups',
        'gl_SubgroupID',
        'gl_SubgroupSize',
        'gl_SubgroupInvocationID',
        'gl_SubgroupEqMask',
        'gl_SubgroupGeMask',
        'gl_SubgroupGtMask',
        'gl_SubgroupLeMask',
        'gl_SubgroupLtMask',
    ]);

    return glsl;
}

monaco.languages.register({id: 'glsl'});
monaco.languages.setMonarchTokensProvider('glsl', definition());
monaco.languages.setLanguageConfiguration('glsl', cpp.conf);

export {};
