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

import $ from 'jquery';
import * as monaco from 'monaco-editor';

import { Alert } from './alert';
import { getStoredSettings } from './settings';
import { FormatRequestOptions } from './formatter-registry.interfaces';
import { SiteSettings } from './settings.interfaces';

const getFormattedCode = ({ source, formatterId, base, tabWidth, useSpaces }: FormatRequestOptions): Promise<string> => {
    const alert = new Alert();
    return new Promise((resolve, reject) => {
        $.ajax(({
            type: 'POST',
            url: `${window.location.origin}${window.httpRoot}api/format/${formatterId}`,
            dataType: 'json',
            contentType: 'application/json',
            data: JSON.stringify({
                source,
                base,
                tabWidth,
                useSpaces,
            }),
            success: (result) => {
                if (result.exit === 0) {
                    resolve(result.answer);
                } else {
                    alert.notify(`We encountered an error formatting your code: ${result.answer}`, {
                        group: 'formatting',
                        alertClass: 'notification-error',
                    });
                }
            },
            error: (xhr, status, error) => {
                // Hopefully we have not exploded!
                if (xhr.responseText) {
                    try {
                        var res = JSON.parse(xhr.responseText);
                        error = res.answer || error;
                    } catch (e) {
                        // continue regardless of error
                    }
                }
                error = error || 'Unknown error';
                alert.notify(`We ran into some issues while formatting your code: ${error}`, {
                    group: 'formatting',
                    alertClass: 'notification-error',
                });
                reject();
            },
            cache: true,
        }));
    });
}

const getDocumentFormatter = (
    language: string,
    formatter: string,
    formatBase: string
): monaco.languages.DocumentFormattingEditProvider => ({
    async provideDocumentFormattingEdits(
        model: monaco.editor.ITextModel,
        options: monaco.languages.FormattingOptions,
        token: monaco.CancellationToken
    ): Promise<monaco.languages.TextEdit[]> {
        const settings: SiteSettings = getStoredSettings();
        const formattedSource = await getFormattedCode({
            source: model.getValue(),
            formatterId: formatter,
            base: formatBase,
            tabWidth: settings.tabWidth,
            useSpaces: settings.useSpaces
        });
        return [{
            range: model.getFullModelRange(),
            text: formattedSource,
        }];
    }
});

monaco.languages.registerDocumentFormattingEditProvider('cppp', getDocumentFormatter('cppp', 'clangformat', 'Google'));
monaco.languages.registerDocumentFormattingEditProvider('nc', getDocumentFormatter('nc', 'clangformat', 'Google'));
monaco.languages.registerDocumentFormattingEditProvider('rust', getDocumentFormatter('rust', 'rustfmt', 'None'));
