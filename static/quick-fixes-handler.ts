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

import _ from 'underscore';
import * as monaco from 'monaco-editor';

type RegisteredQuickFixes = {
    compilerId: number;
    editorModel: monaco.editor.ITextModel;
    fixes: monaco.languages.CodeAction[];
};

let registeredQuickFixes: RegisteredQuickFixes[] = [];
const providersPerLanguage: Record<string, monaco.IDisposable> = {};

export function registerQuickFixesForCompiler(
    compilerId: number,
    editorModel: monaco.editor.ITextModel,
    fixes: monaco.languages.CodeAction[]
): void {
    const item = _.find(registeredQuickFixes, (item: RegisteredQuickFixes): boolean => {
        return item.compilerId === compilerId;
    });

    if (item) {
        item.fixes = fixes;
    } else {
        registeredQuickFixes.push({
            compilerId: compilerId,
            editorModel: editorModel,
            fixes: fixes,
        });
    }
}

function provide(
    model: monaco.editor.ITextModel,
    range: monaco.Range,
    context: monaco.languages.CodeActionContext
): monaco.languages.CodeActionList {
    const item = _.find(registeredQuickFixes, (item: RegisteredQuickFixes): boolean => {
        return item.editorModel === model;
    });

    if (item) {
        return {
            actions: item.fixes.filter(f =>
                f.diagnostics?.some(d =>
                    context.markers.some(m => {
                        const diagnostic = _.pick(
                            d,
                            'message',
                            'startLineNumber',
                            'startColumn',
                            'endLineNumber',
                            'endColumn'
                        );
                        const marker = _.pick(
                            m,
                            'message',
                            'startLineNumber',
                            'startColumn',
                            'endLineNumber',
                            'endColumn'
                        );
                        return _.isEqual(marker, diagnostic);
                    })
                )
            ),
            dispose: function () {},
        };
    } else {
        return {
            actions: [],
            dispose: function () {},
        };
    }
}

export function unregister(compilerId: number): void {
    const item = _.find(registeredQuickFixes, (item: RegisteredQuickFixes): boolean => {
        return item.compilerId === compilerId;
    });

    if (item) {
        registeredQuickFixes = _.without(registeredQuickFixes, item);
    }
}

export function registerProviderForLanguage(language: string): void {
    if (!(language in providersPerLanguage)) {
        providersPerLanguage[language] = monaco.languages.registerCodeActionProvider(language, {
            provideCodeActions: provide,
        });
    }
}
