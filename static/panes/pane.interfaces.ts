// Copyright (c) 2021, Compiler Explorer Authors
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

import * as monaco from 'monaco-editor';

/**
 * The base state of a pane as encoded in the URL.
 *
 * Be aware this state, and any derived state is part of the public API of
 * Compiler Explorer, so don't rename or add anything without careful thought.
 */
export type PaneState = {
    id: number;
    compilerName: string;
    // editorid and treeid are truthy numbers: if they are truthy, then they
    // represent the positive integer id associated with them. If not truthy
    // there is no editor or tree view associated with this pane.
    editorid?: number;
    treeid?: number;
};

// TODO(supergrecko): get the full type
/**
 * The state of a pane that includes basic Monaco editor support.
 *
 * See MonacoPane.
 */
export type MonacoPaneState = PaneState & {
    selection: monaco.Selection | undefined;
};

export type PaneCompilerState = {
    compilerId: number;
    compilerName: string;
    editorId?: number;
    treeId?: number;
};
