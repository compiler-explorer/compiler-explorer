// Copyright (c) 2026, Compiler Explorer Authors
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
    const metal = $.extend(true, {}, cpp.language);

    function addKeywords(keywords: string[]) {
        for (const kw of keywords) {
            metal.keywords.push(kw);
        }
    }

    function vectorTypes(basename: string) {
        const types: string[] = [];
        for (let i = 2; i <= 4; ++i) {
            types.push(`${basename}${i}`);
        }
        return types;
    }

    // Scalar types
    addKeywords(['half', 'ushort', 'uint', 'ulong', 'uchar']);

    // Vector types
    addKeywords(vectorTypes('half'));
    addKeywords(vectorTypes('float'));
    addKeywords(vectorTypes('int'));
    addKeywords(vectorTypes('uint'));
    addKeywords(vectorTypes('short'));
    addKeywords(vectorTypes('ushort'));
    addKeywords(vectorTypes('char'));
    addKeywords(vectorTypes('uchar'));
    addKeywords(vectorTypes('long'));
    addKeywords(vectorTypes('ulong'));
    addKeywords(vectorTypes('bool'));

    // Address space qualifiers
    addKeywords(['device', 'constant', 'threadgroup', 'threadgroup_imageblock', 'thread', 'ray_data', 'object_data']);

    // Function qualifiers
    addKeywords(['kernel', 'vertex', 'fragment', 'tile', 'object', 'mesh', 'visible', 'stitchable', 'extern']);

    // Attribute keywords
    addKeywords([
        'attribute',
        'buffer',
        'texture',
        'sampler',
        'color',
        'position',
        'point_size',
        'clip_distance',
        'cull_distance',
        'render_target_array_index',
        'viewport_array_index',
        'vertex_id',
        'instance_id',
        'base_vertex',
        'base_instance',
        'thread_index_in_threadgroup',
        'thread_position_in_threadgroup',
        'thread_position_in_grid',
        'threadgroup_position_in_grid',
        'threads_per_threadgroup',
        'threads_per_grid',
        'dispatch_quadgroups_per_threadgroup',
        'dispatch_simdgroups_per_threadgroup',
        'quadgroup_index_in_threadgroup',
        'simdgroup_index_in_threadgroup',
        'thread_index_in_quadgroup',
        'thread_index_in_simdgroup',
        'front_facing',
        'primitive_id',
        'barycentric_coord',
        'depth',
        'stencil',
        'sample_id',
        'sample_mask',
        'cover_value',
        'rasterization_rate_map_data',
        'amplification_count',
        'amplification_id',
        'payload',
        'mesh',
        'stage_in',
    ]);

    // Texture types
    addKeywords([
        'texture1d',
        'texture1d_array',
        'texture2d',
        'texture2d_array',
        'texture2d_ms',
        'texture2d_ms_array',
        'texture3d',
        'texturecube',
        'texturecube_array',
        'depth2d',
        'depth2d_array',
        'depth2d_ms',
        'depth2d_ms_array',
        'depthcube',
        'depthcube_array',
        'texture_buffer',
    ]);

    // Sampler types
    addKeywords(['sampler', 'sampler_state', 'min_filter', 'mag_filter', 'mip_filter', 'address', 'compare_func']);

    // Matrix types
    addKeywords([
        'float2x2',
        'float2x3',
        'float2x4',
        'float3x2',
        'float3x3',
        'float3x4',
        'float4x2',
        'float4x3',
        'float4x4',
        'half2x2',
        'half2x3',
        'half2x4',
        'half3x2',
        'half3x3',
        'half3x4',
        'half4x2',
        'half4x3',
        'half4x4',
    ]);

    // Built-in functions
    addKeywords([
        'discard_fragment',
        'simd_vote',
        'simd_ballot',
        'simd_all',
        'simd_any',
        'simd_sum',
        'simd_product',
        'simd_min',
        'simd_max',
        'simd_prefix_inclusive_sum',
        'simd_prefix_exclusive_sum',
        'simd_prefix_inclusive_product',
        'simd_prefix_exclusive_product',
        'simd_broadcast',
        'simd_broadcast_first',
        'simd_shuffle',
        'simd_shuffle_xor',
        'simd_shuffle_rotate_up',
        'simd_shuffle_rotate_down',
        'quad_vote',
        'quad_broadcast',
        'quad_shuffle',
        'quad_shuffle_xor',
        'quad_shuffle_rotate_up',
        'quad_shuffle_rotate_down',
    ]);

    return metal;
}

monaco.languages.register({id: 'metal'});
monaco.languages.setMonarchTokensProvider('metal', definition());
monaco.languages.setLanguageConfiguration('metal', cpp.conf);
