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

import {describe, expect, it, vi} from 'vitest';

import {CompilerInfo} from '../../../types/compiler.interfaces.js';
import {DucklingOutputView, formatDucklingParserOutput} from '../../panes/duckling-output-view.js';

describe('Duckling parser output formatting', () => {
    it('expands Duckling’s JSON-shaped AST and removes its trailing commas', () => {
        const result = formatDucklingParserOutput(
            'Parsed tree:\n{"position":{"sourceStart":0},"items":[{"name":"main",},],}\n',
        );

        expect(result).toEqual({
            output: '{\n  "position": {\n    "sourceStart": 0\n  },\n  "items": [\n    {\n      "name": "main"\n    }\n  ]\n}',
            languageId: 'json',
        });
    });

    it('preserves diagnostics and other non-JSON parser output as plain text', () => {
        expect(formatDucklingParserOutput('parser failed')).toEqual({
            output: 'parser failed',
            languageId: 'plaintext',
        });
    });
});

describe('Duckling output view compiler capabilities', () => {
    function viewWithOutputKind(outputKind: string) {
        const view = Object.create(DucklingOutputView.prototype) as DucklingOutputView;
        const eventHub = {emit: vi.fn()};
        const showMessage = vi.fn();
        Object.assign(view, {
            compilerInfo: {compilerId: 12, compilerName: '', editorId: 3, treeId: 0},
            outputKind,
            outputViewOpenAnnounced: false,
            eventHub,
            showMessage,
            updateTitle: vi.fn(),
            updateState: vi.fn(),
        });
        return {view, eventHub, showMessage};
    }

    it('announces a pane with its fixed output kind when the compiler supports it', () => {
        const {view, eventHub} = viewWithOutputKind('hir');
        const compiler = {name: 'Duckling DVM', supportsDucklingOutputs: ['hir', 'mir']} as CompilerInfo;

        view.onCompiler(12, compiler, '', 3, 0);

        expect(eventHub.emit).toHaveBeenCalledWith('ducklingOutputViewOpened', 12, 'hir');
    });

    it('explains when the selected compiler does not support this pane kind', () => {
        const {view, showMessage} = viewWithOutputKind('parser');
        const compiler = {name: 'C++ compiler', supportsDucklingOutputs: []} as unknown as CompilerInfo;

        view.onCompiler(12, compiler, '', 3, 0);

        expect(showMessage).toHaveBeenCalledWith('<Duckling outputs are not supported for this compiler>');
    });
});
