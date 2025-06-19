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

import {parse} from '../shared/stacktrace.js';

import {options} from './options.js';

import * as Sentry from '@sentry/browser';

import GoldenLayout from 'golden-layout';
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
                /this.error\(new CancellationError\(\)/,
                /new StandardMouseEvent\(monaco-editor/,
                /Object Not Found Matching Id:2/,
                /Illegal value for lineNumber/,
                'SlowRequest',
            ],
            beforeSend(event, hint) {
                // Filter Monaco Editor errors
                // NOTE: When debugging filters, check the actual JSON structure from Sentry UI
                // - event.exception.values[0].value contains the error message
                // - event.exception.values[0].type contains the error type
                // - Sentry UI shows "type: value" but actual JSON has separate fields
                // - frames[0] is often Sentry's wrapper, not the original error source
                if (event.exception?.values?.[0]?.stacktrace?.frames) {
                    const frames = event.exception.values[0].stacktrace.frames;

                    // Filter hit testing errors
                    // See: https://github.com/microsoft/monaco-editor/issues/4527
                    // NOTE: Function name appears in error message, not as frame.function due to Sentry wrapping
                    if (event.exception.values[0].value?.includes('_doHitTestWithCaretPositionFromPoint')) {
                        return null; // Don't send to Sentry
                    }

                    // Filter clipboard cancellation errors
                    // NOTE: Frame filename matching works as expected for file path filtering
                    const hasClipboardFrame = frames.some(frame =>
                        frame.filename?.includes('monaco-editor/esm/vs/platform/clipboard/browser/clipboardService.js'),
                    );
                    // NOTE: Error value may be "Canceled" or "Canceled\nSentryCapture Context: ..." due to context addition
                    const isCancellationError = event.exception.values[0].value?.startsWith('Canceled');

                    if (hasClipboardFrame && isCancellationError) {
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
