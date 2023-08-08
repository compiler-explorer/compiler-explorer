// Copyright (c) 2022, Compiler Explorer Authors
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
import * as cpp from 'monaco-editor/esm/vs/basic-languages/cpp/cpp';

function definition(): monaco.languages.IMonarchLanguage {
    const hlsl = $.extend(true, {}, cpp.language);

    function addKeywords(keywords) {
        for (let i = 0; i < keywords.length; ++i) {
            hlsl.keywords.push(keywords[i]);
        }
    }

    function vectorMatrixTypes(basename) {
        const types: string[] = [];
        for (let i = 1; i !== 5; ++i) {
            for (let j = 1; j !== 5; ++j) {
                types.push(`${basename}${i}x${j}`);
            }
            types.push(`${basename}${i}`);
        }
        return types;
    }

    addKeywords(vectorMatrixTypes('bool'));
    addKeywords(vectorMatrixTypes('uint'));
    addKeywords(vectorMatrixTypes('float'));
    addKeywords(vectorMatrixTypes('int'));

    function resource(name) {
        return [name, `RW${name}`];
    }

    addKeywords(resource('Buffer'));
    addKeywords(resource('Texture1D'));
    addKeywords(resource('Texture1DArray'));
    addKeywords(resource('Texture2D'));
    addKeywords(resource('Texture2DArray'));
    addKeywords(resource('Texture3D'));
    addKeywords(resource('TextureCube'));
    addKeywords(resource('TextureCubeArray'));
    addKeywords(resource('Texture2DMS'));
    addKeywords(resource('Texture2DMSArray'));
    addKeywords(resource('ByteAddressBuffer'));
    addKeywords(resource('StructuredBuffer'));
    addKeywords(resource('ConstantBuffer'));

    addKeywords([
        'out',
        'inout',
        'vector',
        'matrix',
        'uint',

        'SamplerState',
        'SamplerComparisonState',

        // Additional resource yypes
        'AppendStructuredBuffer',
        'ConsumeStructuredBuffer',

        // Intrinsic functions
        'abort',
        'abs',
        'acos',
        'all',
        'AllMemoryBarrier',
        'AllMemoryBarrierWithGroupSync',
        'any',
        'asdouble',
        'asfloat',
        'asin',
        'asint',
        'asuint',
        'atan',
        'atan2',
        'ceil',
        'CheckAccessFullyMapped',
        'clamp',
        'clip',
        'cos',
        'cosh',
        'countbits',
        'cross',
        'ddx',
        'ddx_coarse',
        'ddx_fine',
        'ddy',
        'ddy_coarse',
        'ddy_fine',
        'degrees',
        'determinant',
        'DeviceMemoryBarrier',
        'DeviceMemoryBarrierWithGroupSync',
        'distance',
        'dot',
        'dst',
        'errorf',
        'EvaluateAttributeCentroid',
        'EvaluateAttributeAtSample',
        'EvaluateAttributeSnapped',
        'exp',
        'exp2',
        'f16tof32',
        'f32tof16',
        'faceforward',
        'firstbithigh',
        'firstbitlow',
        'floor',
        'fma',
        'fmod',
        'frac',
        'frexp',
        'fwidth',
        'GetRenderTargetSampleCount',
        'GetRenderTargetSamplePosition',
        'GroupMemoryBarrier',
        'GroupMemoryBarrierWithGroupSync',
        'InterlockedAdd',
        'InterlockedAnd',
        'InterlockedCompareExchange',
        'InterlockedExchange',
        'InterlockedMax',
        'InterlockedMin',
        'InterlockedOr',
        'InterlockedXor',
        'isfinite',
        'isinf',
        'isnan',
        'ldexp',
        'length',
        'lerp',
        'lit',
        'log',
        'log10',
        'log2',
        'mad',
        'max',
        'min',
        'modf',
        'msad4',
        'mul',
        'noise',
        'normalize',
        'pow',
        'Process2DQuadTessFactorsAvg',
        'Process2DQuadTessFactorsMax',
        'Process2DQuadTessFactorsMin',
        'ProcessIsolineTessFactors',
        'ProcessQuadTessFactorsAvg',
        'ProcessQuadTessFactorsMax',
        'ProcessQuadTessFactorsMin',
        'ProcessTriTessFactorsAvg',
        'ProcessTriTessFactorsMax',
        'ProcessTriTessFactorsMin',
        'radians',
        'rcp',
        'reflect',
        'refract',
        'reversebits',
        'round',
        'rsqrt',
        'saturate',
        'sign',
        'sin',
        'sincos',
        'sinh',
        'smoothstep',
        'sqrt',
        'step',
        'tan',
        'tanh',
        'tex1D',
        'tex1Dbias',
        'tex1Dgrad',
        'tex1Dlod',
        'tex1Dproj',
        'tex2D',
        'tex2Dbias',
        'tex2Dgrad',
        'tex2Dlod',
        'tex2Dproj',
        'tex3D',
        'tex3Dbias',
        'tex3Dgrad',
        'tex3Dlod',
        'tex3Dproj',
        'texCUBE',
        'texCUBEbias',
        'texCUBEgrad',
        'texCUBElod',
        'texCUBEproj',
        'transpose',
        'trunc',

        // Wave intrinsics
        'WaveIsFirstLane',
        'WaveGetLaneCount',
        'WaveGetLaneIndex',
        'WaveActiveAnyTrue',
        'WaveActiveAllTrue',
        'WaveActiveBallot',
        'WaveReadLaneFirst',
        'WaveReadLaneAt',
        'WaveActiveAllEqual',
        'WaveActiveCountBits',
        'WaveActiveSum',
        'WaveActiveProduct',
        'WaveActiveBitAnd',
        'WaveActiveBitOr',
        'WaveActiveBitXor',
        'WaveActiveMin',
        'WaveActiveMax',
        'WavePrefixCountBits',
        'WavePrefixProduct',
        'WavePrefixSum',
        'QuadReadAcrossX',
        'QuadReadAcrossY',
        'QuadReadAcrossDiagonal',
        'QuadReadLaneAt',
        // SM6.5 wave intrinsics
        'WaveMatch',
        'WaveMultiPrefixSum',
        'WaveMultiPrefixProduct',
        'WaveMultiPrefixCountBits',
        'WaveMultiPrefixAnd',
        'WaveMultiPrefixOr',
        'WaveMultiPrefixXor',

        // Raytracing intrinsics
        'RayTracingAccelerationStructure',
        'AcceptHitAndEndSearch',
        'CallShader',
        'IgnoreHit',
        'PrimitiveIndex',
        'ReportHit',
        'TraceRay',
        'DispatchRaysIndex',
        'DispatchRaysDimensions',
        'WorldRayOrigin',
        'WorldRayDirection',
        'RayTMin',
        'RayTCurrent',
        'RayFlags',
        'InstanceIndex',
        'InstanceID',
        'ObjectRayOrigin',
        'ObjectRayDirection',
        'ObjectToWorld3x4',
        'ObjectToWorld4x3',
        'WorldToObject3x4',
        'WorldToObject4x3',
        'HitKind',

        // SM6.4
        'dot4add_u8packed',
        'dot4add_i8packed',
        'dot2add',

        // Compute + mesh/amplification shaders
        'numthreads',
        'outputtopology',
        'DispatchMesh',
        'groupshared',

        // SM6.6
        'InterlockedCompareStore',
        'InterlockedCompareStoreFloatBitwise',
        'InterlockedCompareExchangeFloatBitwise',
        'ResourceDescriptorHeap',
        'SamplerDescriptorHeap',
        'IsHelperLane',
        'uint8_t4_packed',
        'int8_t4_packed',
        'unpack_s8s16',
        'unpack_u8u16',
        'unpack_s8s32',
        'unpack_u8u32',
        'pack_u8',
        'pack_s8',
        'pack_clamp_u8',
        'pack_clamp_s8',
        'WaveSize',
        'raypayload',

        // SM6.7
        'QuadAll',
        'QuadAny',
        'WaveOpsIncludeHelperLanes',
    ]);

    return hlsl;
}

monaco.languages.register({id: 'hlsl'});
monaco.languages.setMonarchTokensProvider('hlsl', definition());

export {};
