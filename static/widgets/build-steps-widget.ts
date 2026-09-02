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

import {BuildStep} from '../../types/compilation/compilation.interfaces.js';
import * as BootstrapUtils from '../bootstrap-utils.js';
import {CompilationStatusCode} from '../compiler-service.interfaces.js';
import {CompilerService} from '../compiler-service.js';

// Quotes only what a shell would need it quoted for, so the common case stays readable and the
// line can still be pasted into a terminal as-is.
function shellQuote(arg: string): string {
    return /^[\w.,:=@%+/-]*$/.test(arg) ? arg : `'${arg.replace(/'/g, `'\\''`)}'`;
}

function commandLineOf(step: BuildStep): string {
    return [step.command, ...step.compilationOptions]
        .filter(Boolean)
        .map(arg => shellQuote(arg!))
        .join(' ');
}

function renderEnv(env: Record<string, string>): JQuery<HTMLElement> {
    const table = $('<table/>').addClass('build-step-env');
    for (const key of Object.keys(env).sort()) {
        const row = $('<tr/>');
        row.append($('<td/>').addClass('build-step-env-key').text(key));
        row.append($('<td/>').addClass('build-step-env-value').text(env[key]));
        table.append(row);
    }
    return table;
}

function renderStep(step: BuildStep): JQuery<HTMLElement> {
    const container = $('<div/>').addClass('build-step');

    const header = $('<div/>').addClass('build-step-header');
    const icon = $('<span/>').addClass('status-icon');
    // Same status language as everywhere else, so the step that failed is the one wearing the red cross.
    // Judged on the exit code alone rather than via calculateStatusIcon: that treats any output at all as a
    // warning, which suits a compiler but not a build tool, since cmake narrates its progress on stdout and
    // would leave every successful step looking orange.
    const code = step.code === 0 ? CompilationStatusCode.OK : CompilationStatusCode.WITH_ERRORS;
    CompilerService.handleCompilationStatus(null, icon, {code, compilerOut: step.code});
    header.append(icon);
    header.append($('<span/>').addClass('build-step-name').text(step.step));
    if (step.execTime) header.append($('<span/>').addClass('build-step-time').text(`${step.execTime}ms`));
    container.append(header);

    container.append($('<pre/>').addClass('build-step-command').text(commandLineOf(step)));

    const env = step.env;
    if (env && Object.keys(env).length > 0) {
        const details = $('<details/>').addClass('build-step-env-details');
        details.append($('<summary/>').text(`Environment (${Object.keys(env).length})`));
        details.append(renderEnv(env));
        container.append(details);
    }

    return container;
}

export function displayBuildSteps(buildsteps: BuildStep[]): void {
    const modal = $('#build-steps');
    const body = modal.find('.build-steps-body');
    body.empty();

    if (buildsteps.length === 0) {
        body.append($('<p/>').text('This compilation ran no build steps.'));
    } else {
        for (const step of buildsteps) body.append(renderStep(step));
    }

    BootstrapUtils.showModal(modal);
}
