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
import _ from 'underscore';

import {escapeHTML, maskRootdirKeepingAppPrefix} from '../../shared/common-utils.js';
import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import * as BootstrapUtils from '../bootstrap-utils.js';
import {CompilationStatusCode} from '../compiler-service.interfaces.js';
import {CompilerService} from '../compiler-service.js';
import {Pane} from '../panes/pane.js';
import {Toggles} from './toggles.js';

type GetResult = (result: CompilationResult) => Pick<CompilationResult, 'code' | 'compilationOptions'> | undefined;

abstract class BaseCompilationOptions<P extends Pane<object>> {
    protected readonly parent: P;
    private readonly parentId: number;
    private readonly prependOptions: JQuery<HTMLElement>;
    private readonly statusIcon: JQuery<HTMLElement>;
    private readonly getResult: GetResult;
    private readonly filters?: Toggles;

    constructor(pane: P, parentId: number, getResult: GetResult, filters?: Toggles) {
        this.parent = pane;
        this.parentId = parentId;
        this.prependOptions = pane.domRoot.find('.prepend-options');
        this.statusIcon = this.prependOptions.find('.status-icon');
        this.getResult = getResult;
        this.filters = filters;
        this.initCallbacks();

        $(document).on('mouseup', e => {
            const target = $(e.target);
            if (
                !target.is(this.prependOptions) &&
                this.prependOptions.has(target as unknown as Element).length === 0 &&
                target.closest('.popover').length === 0
            ) {
                BootstrapUtils.hidePopover(this.prependOptions);
            }
        });
    }

    protected initCallbacks(): void {}

    protected onCompiling(compilerId: number, compiler: CompilerInfo): void {
        if (this.parentId !== compilerId) return;
        // Display the spinner
        CompilerService.handleCompilationStatus(this.statusIcon, {
            code: CompilationStatusCode.COMPILING,
            compilerOut: 0,
        });
    }

    protected onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void {
        this.handleResult(compilerId, compiler, result);
    }

    protected onExecuteResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void {
        this.handleResult(compilerId, compiler, result, result.didExecute);
    }

    private handleResult(
        compilerId: number,
        compiler: CompilerInfo,
        result: CompilationResult,
        didExecute?: boolean,
    ): void {
        if (this.parentId !== compilerId) return;

        const wasCmake = result.result ? (result.buildsteps?.some(step => step.step === 'cmake') ?? false) : false;
        const stepResult = this.getResult(result);
        CompilerService.handleCompilationStatus(this.statusIcon, {
            didExecute,
            ...CompilerService.calculateStatusIcon(stepResult ?? result),
        });
        const options = (stepResult?.compilationOptions ?? []).map(maskRootdirKeepingAppPrefix);
        this.setCompilationOptionsPopover(options.join(' '), this.checkForUnwiseArguments(compiler, options, wasCmake));
    }

    private setCompilationOptionsPopover(content: string | null, warnings: string[]): void {
        const infoLine =
            '<div class="compiler-arg-warning info">You can configure icon animations in Settings>Compilation</div>\n';

        // Dispose any existing popover
        const existingPopover = BootstrapUtils.getPopoverInstance(this.prependOptions);
        if (existingPopover) existingPopover.dispose();

        // Create new popover
        BootstrapUtils.initPopover(this.prependOptions, {
            content:
                warnings.map(w => `<div class="compiler-arg-warning">${w}</div>`).join('\n') +
                '\n' +
                (warnings.length > 0 ? infoLine : '') +
                escapeHTML(content || 'No options in use') +
                `\n<div class="compiler-arg-warning-shake-setting"></div>`,
            html: true,
            template:
                '<div class="popover' +
                (content ? ' compiler-options-popover' : '') +
                '" role="tooltip"><div class="arrow"></div>' +
                '<h3 class="popover-header"></h3><div class="popover-body"></div></div>',
        });

        // TODO: Kind of redundant with compiler-service's handleCompilationStatus and overriding what that function
        // does. I hate that the logic is spread out like this. Definitely in need of a refactor.
        if (warnings.length > 0) {
            this.statusIcon
                .removeClass()
                .addClass(
                    'status-icon fa-solid fa-triangle-exclamation compiler-arg-warning-icon' +
                        (this.parent.settings.shakeStatusIconOnWarnings ? ' shake' : ''),
                )
                .css('color', '')
                .attr('aria-label', 'There are warnings about the compiler arguments that have been provided');
        }
    }

    private checkForUnwiseArguments(compiler: CompilerInfo, optionsArray: string[], wasCmake: boolean) {
        // Check if any options are in the unwiseOptions array and remember them
        const unwiseOptions = _.intersection(
            optionsArray,
            compiler.unwiseOptions.filter(opt => opt !== ''),
        );

        const options = unwiseOptions.length === 1 ? 'Option ' : 'Options ';
        const names = unwiseOptions.join(', ');
        const are = unwiseOptions.length === 1 ? ' is ' : ' are ';
        const msg = options + names + are + 'not recommended, as behaviour might change based on server hardware.';

        const warnings: string[] = [];

        if (optionsArray.some(opt => opt === '-flto') && !(this.filters?.isSet('binary') ?? true) && !wasCmake) {
            warnings.push('Option -flto is being used without Link to Binary.');
        }

        if (unwiseOptions.length > 0) {
            warnings.push(msg);
        }

        return warnings;
    }
}

export class CompilationOptions<P extends Pane<object>> extends BaseCompilationOptions<P> {
    protected override initCallbacks(): void {
        super.initCallbacks();
        this.parent.eventHub.on('compiling', this.onCompiling.bind(this));
        this.parent.eventHub.on('compileResult', this.onCompileResult.bind(this));
    }
}

export class ExecutionCompilationOptions<P extends Pane<object>> extends BaseCompilationOptions<P> {
    protected override initCallbacks(): void {
        super.initCallbacks();
        this.parent.eventHub.on('executeCompiling', this.onCompiling.bind(this));
        this.parent.eventHub.on('executeResult', this.onExecuteResult.bind(this));
    }
}
