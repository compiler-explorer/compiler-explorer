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

import {Alert} from './alert';
import {Settings} from './settings';
import {FormattingRequest} from './api/formatting.interfaces';
import {getFormattedCode} from './api/api';

// Proxy function to emit the error to the alert system
const onFormatError = (cause: string, source: string) => {
    const alertSystem = new Alert();
    alertSystem.notify(`We encountered an error formatting your code: ${cause}`, {
        group: 'formatting',
        alertClass: 'notification-error',
    });
    return source;
};

const doFormatRequest = async (options: FormattingRequest) => {
    const res = await getFormattedCode(options);
    const body = await res.json();
    if (res.status === 200 && body.exit === 0) {
        // API sent 200 and we have a valid response
        return body.answer as string;
    }
    // We had an error (either HTTP request error, or API error)
    // Figure out which it is, show it to the user, and reject the promise
    const cause = body.answer ?? res.statusText;
    throw new Error(cause);
};

/**
 * Create a monaco DocumentFormattingEditProvider for a registered monaco
 * language.
 *
 * @param language - The monaco-editor registered language to format code for
 * @param formatter - The CE format API backend to use
 * @param isOneTrueStyle - Whether the CE format API backend has one true style
 */
const getDocumentFormatter = (
    language: string,
    formatter: string,
    isOneTrueStyle: boolean
): monaco.languages.DocumentFormattingEditProvider => ({
    async provideDocumentFormattingEdits(
        model: monaco.editor.ITextModel,
        options: monaco.languages.FormattingOptions,
        token: monaco.CancellationToken
    ): Promise<monaco.languages.TextEdit[]> {
        const settings = Settings.getStoredSettings();
        // If there is only one style, return __DefaultStyle.
        const base = isOneTrueStyle ? '__DefaultStyle' : settings.formatBase;
        const source = model.getValue();
        // Request the formatted code. If that API call fails, we just back off
        // and return the user's old code.
        const formattedSource = await doFormatRequest({
            formatterId: formatter,
            tabWidth: settings.tabWidth,
            useSpaces: settings.useSpaces,
            source,
            base,
        }).catch(err => onFormatError(err, source));
        return [
            {
                range: model.getFullModelRange(),
                text: formattedSource,
            },
        ];
    },
});

/** Register a monaco-editor language's default document formatting provider */
const register = (lang: string, formatter: string, isOneTrueStyle: boolean) => {
    const provider = getDocumentFormatter(lang, formatter, isOneTrueStyle);
    monaco.languages.registerDocumentFormattingEditProvider(lang, provider);
};

register('cppp', 'clangformat', false);
register('nc', 'clangformat', false);
register('go', 'gofmt', true);
register('rust', 'rustfmt', true);
register('dart', 'dartformat', true);
