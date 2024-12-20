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
    const slang = $.extend(true, {}, cpp.language);

    function addKeywords(keywords: string[]) {
        for (let i = 0; i < keywords.length; ++i) {
            slang.keywords.push(keywords[i]);
        }
    }

    function vectorMatrixTypes(basename: string) {
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

    function resource(name: string) {
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
        // Additional resource types
        'SamplerState',
        'SamplerComparisonState',
        'AppendStructuredBuffer',
        'ConsumeStructuredBuffer',
        'ParameterBlock',
        'Ptr',
        'GLSLShaderStorageBuffer',
        'Atomic',
        'WTexture2D',

        // non-C style statement words
        'throws',
        'in',
        'out',
        'inout',
        'vector',
        'matrix',
        'uint',
        'ref',
        '__subscript',
        '__init',
        'get',
        'set',
        'typealias',
        'uniform',
        'groupshared',
        'extension',
        'associatedtype',
        'This',
        '__generic',
        '__exported',
        'import',
        'discard',
        'defer',
        'cbuffer',
        'tbuffer',
        'func',
        'is',
        'as',
        'none',
        'functype',
        'alignof',
        '__target_switch',
        '__intrinsic_asm',
        'each',
        'expand',
        'let',
        'var',
        'no_diff',
        'dynamic_uniform',
        'fwd_diff',
        'bwd_diff',
        'module',
        'implementing',
        '__include',
        '__dispatch_kernel',
        'row_major',
        'column_major',
        'nointerpolation',
        'snorm',
        'unorm',
        'globallycoherent',
        'layout',
        'numthreads',

        // Intrinsic functions
        'abort',
        'abs',
        'acos',
        'all',
        'any',
        'asin',
        'asint',
        'asuint',
        'atan',
        'atan2',
        'ceil',
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
        'distance',
        'dot',
        'dst',
        'errorf',
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
        'transpose',
        'trunc',

        // vulkan semantic names
        'vk',
        'binding',
        'shader',
        'spirv_asm',

        // HLSL semantic names
        'register',
        'packoffset',
        'read',
        'write',
        'SV_ClipDistance',
        'SV_CullDistance',
        'SV_Coverage',
        'SV_Depth',
        'SV_DepthGreaterEqual',
        'SV_DepthLessEqual',
        'SV_DispatchThreadID',
        'SV_DomainLocation',
        'SV_GroupID',
        'SV_GroupIndex',
        'SV_GroupThreadID',
        'SV_GSInstanceID',
        'SV_InnerCoverage',
        'SV_InsideTessFactor',
        'SV_InstanceID',
        'SV_IsFrontFace',
        'SV_OutputControlPointID',
        'SV_Position',
        'SV_PointSize',
        'SV_PrimitiveID',
        'SV_RenderTargetArrayIndex',
        'SV_SampleIndex',
        'SV_StencilRef',
        'SV_Target',
        'SV_TessFactor',
        'SV_VertexID',
        'SV_ViewID',
        'SV_ViewportArrayIndex',
        'SV_ShadingRate',
    ]);

    return slang;
}

monaco.languages.register({id: 'slang'});
monaco.languages.setMonarchTokensProvider('slang', definition());
monaco.languages.setLanguageConfiguration('slang', cpp.conf);

export {};
