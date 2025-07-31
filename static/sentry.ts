// Copyright (c) 2023, Compiler Explorer Authors
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

import * as Sentry from '@sentry/browser';
import GoldenLayout from 'golden-layout';
import {parse} from '../shared/stacktrace.js';
import {options} from './options.js';
import {SiteSettings} from './settings.js';
import {serialiseState} from './url.js';

let layout: GoldenLayout;
let allowSendCode: boolean;

export function setSentryLayout(l: GoldenLayout) {
    layout = l;
    layout.eventHub.on('settingsChange', (newSettings: SiteSettings) => {
        allowSendCode = newSettings.allowStoreCodeDebug;
    });

    Sentry.addEventProcessor(event => {
        if (!allowSendCode) {
            return event;
        }
        try {
            const config = layout.toConfig();
            if (event.extra === undefined) {
                event.extra = {};
            }
            event.extra['full_url'] = window.location.origin + window.httpRoot + '#' + serialiseState(config);
        } catch (e) {
            console.log('Error adding full_url to Sentry event', e);
        }
        return event;
    });
}

export function SetupSentry() {
    if (options.statusTrackingEnabled && options.sentryDsn) {
        Sentry.init({
            dsn: options.sentryDsn,
            release: options.release,
            environment: options.sentryEnvironment,
            ignoreErrors: [
                // NOTE: Monaco Editor patterns may not work reliably due to code minification
                // Source mapping happens AFTER beforeSend/ignoreErrors processing
                /this.error\(new CancellationError\(\)/,
                /new StandardMouseEvent\(monaco-editor/,
                // CEFSharp bot errors - these come from automated scanners, particularly Microsoft Outlook's
                // SafeLink feature that scans URLs in emails. The error format "Object Not Found Matching Id:X,
                // MethodName:Y, ParamCount:Z" is specific to CEFSharp (.NET Chromium wrapper).
                // Analysis of 76,000+ events shows 100% come from Windows + Chrome (CEFSharp's signature).
                // This pattern was previously seen with Id:2 (PR #7103).
                // See: https://github.com/DataDog/browser-sdk/issues/2715
                /Object Not Found Matching Id:\d+/,
                /Illegal value for lineNumber/,
                'SlowRequest',
                // Monaco Editor clipboard cancellation errors
                'Canceled',
            ],
            beforeSend(event, hint) {
                // Filter Monaco Editor errors
                //
                // IMPORTANT: Frame-based filtering doesn't work reliably!
                // In beforeSend hooks, frame.filename contains minified bundle paths like:
                // "https://static.ce-cdn.net/vendor.v59.be68c0bf31258854d1b2.js"
                //
                // Source mapping happens AFTER beforeSend processing, which is why the
                // final Sentry UI shows readable paths like:
                // "./node_modules/monaco-editor/esm/vs/platform/clipboard/browser/clipboardService.js"
                //
                // For reliable filtering, use:
                // 1. ignoreErrors patterns (processed before beforeSend)
                // 2. Error message content (event.exception.values[0].value)
                // 3. Error type (event.exception.values[0].type)
                //
                // DO NOT rely on frame.filename, frame.module, or frame.function for Monaco errors!

                if (event.exception?.values?.[0]) {
                    // Filter hit testing errors
                    // See: https://github.com/microsoft/monaco-editor/issues/4527
                    // Uses error message content since frame data is minified
                    if (event.exception.values[0].value?.includes('_doHitTestWithCaretPositionFromPoint')) {
                        return null; // Don't send to Sentry
                    }
                }
                return event;
            },
        });
        window.addEventListener('unhandledrejection', event => {
            // Convert non-Error rejection reasons to Error objects
            let reason = event.reason;
            if (!(reason instanceof Error)) {
                const errorMessage =
                    typeof reason === 'string' ? reason : `Non-Error rejection: ${JSON.stringify(reason)}`;
                reason = new Error(errorMessage);

                // Preserve original reason for debugging
                (reason as any).originalReason = event.reason;
            }
            SentryCapture(reason, 'Unhandled Promise Rejection');
        });
    }
}

export function SentryCapture(value: unknown, context?: string) {
    if (value instanceof Error) {
        if (context) {
            value.message += `\nSentryCapture Context: ${context}`;
        }
        Sentry.captureException(value);
    } else {
        const e = new Error();
        const trace = parse(e);
        Sentry.captureMessage(
            'Non-Error capture:\n' +
                (context ? `Context: ${context}\n` : '') +
                `Data:\n${JSON.stringify(value)}\n` +
                'Trace:\n' +
                trace
                    .map(frame => `${frame.functionName} ${frame.fileName}:${frame.lineNumber}:${frame.columnNumber}`)
                    .join('\n'),
        );
    }
}
