// Copyright (c) 2025, Compiler Explorer Authors
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

import {CompilationResult} from '../compilation/compilation.interfaces.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {type EventMap} from '../event-map.js';
import {Output} from './output.js';

export class RustClippy extends Output {
    override openEventKind(): keyof EventMap {
        return 'rustClippyViewOpened';
    }

    override closeEventKind(): keyof EventMap {
        return 'rustClippyViewClosed';
    }

    override getDefaultPaneName(): string {
        return 'Rust Clippy Viewer';
    }

    override onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        const {rustClippyOutput, stderr, ...clippyResult} = result;
        if (rustClippyOutput) {
            super.onCompileResult(compilerId, compiler, rustClippyOutput);
        } else if (compiler.supportsRustClippyView) {
            super.onCompileResult(compilerId, compiler, {stderr: [{text: '<No output>'}], ...clippyResult});
        }
    }

    override onCompiler(
        compilerId: number,
        compiler: CompilerInfo | null,
        options: string,
        editorId?: number,
        treeId?: number,
    ): void {
        if (this.compilerInfo.compilerId === compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.compilerInfo.editorId = editorId;
            this.compilerInfo.treeId = treeId;
            this.updateTitle();
            if (compiler && !compiler.supportsRustClippyView) {
                super.onCompileResult(compilerId, compiler, {
                    stdout: [],
                    stderr: [{text: '<Rust Clippy is not supported for this compiler>'}],
                    code: -1,
                    timedOut: false,
                });
            }
        }
    }
}
