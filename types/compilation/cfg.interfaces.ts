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

// TODO(jeremy-rifkin): re-visit all the types here once the back-end is more typescripted

export type EdgeDescriptor = {
    from: string;
    to: string;
    arrows: string; // <- useless
    color: string;
};

export type NodeDescriptor = {
    id: string; // typically label for the bb
    label: string; // really the source
};

export type AnnotatedNodeDescriptor = NodeDescriptor & {
    width: number; // in pixels
    height: number; // in pixels
};

type CfgDescriptor_<ND> = {
    edges: EdgeDescriptor[];
    nodes: ND[];
};

export type CfgDescriptor = CfgDescriptor_<NodeDescriptor>;
export type AnnotatedCfgDescriptor = CfgDescriptor_<AnnotatedNodeDescriptor>;

// function name -> cfg data
export type CFGResult = Record<string, CfgDescriptor>;
export type AnnotatedCFGResult = Record<string, AnnotatedCfgDescriptor>;
