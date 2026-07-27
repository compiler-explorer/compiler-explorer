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

import {escapeHTML} from '../../shared/common-utils.js';
import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import * as BootstrapUtils from '../bootstrap-utils.js';
import {CompilerService} from '../compiler-service.js';
import {EventHub} from '../event-hub.js';
import {Toggles} from '../widgets/toggles.js';
import {PaneCompilerState} from './pane.interfaces.js';
import {Pane} from './pane.js';

type GetResult = (result: CompilationResult) => {compilationOptions?: string[]} | undefined;

export class CompilationOptions<P extends Pane<object>> {
    private readonly parent: P;
    private readonly eventHub: EventHub;
    private readonly prependOptions: JQuery<HTMLElement>;
    private readonly statusIcon: JQuery<HTMLElement>;
    private readonly getResult: GetResult;
    private readonly filters?: Toggles;

    constructor(pane: P, prependOptions: JQuery<HTMLElement>, getResult: GetResult, filters?: Toggles) {
        this.parent = pane;
        this.eventHub = pane.eventHub;
        this.prependOptions = prependOptions;
        this.statusIcon = prependOptions.find('.status-icon');
        this.getResult = getResult;
        this.filters = filters;

        this.eventHub.on('compileResult', this.onCompileResult.bind(this));
        this.eventHub.on('compilerClose', this.close.bind(this));

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

    private get compilerInfo(): PaneCompilerState {
        return this.parent.compilerInfo;
    }

    private close(compilerId: number, sourceTreeId: number | boolean): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        this.eventHub.unsubscribe();
    }

    private onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (this.compilerInfo.compilerId !== compilerId) return;

        const wasCmake = result.result ? (result.buildsteps?.some(step => step.step === 'cmake') ?? false) : false;
        const options = this.getResult(result)?.compilationOptions;
        this.setCompilationOptionsPopover(
            options ? options.join(' ') : '',
            this.checkForUnwiseArguments(compiler, options, wasCmake),
        );
        CompilerService.handleCompilationStatus(null, this.statusIcon, CompilerService.calculateStatusIcon(result));
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

    private checkForUnwiseArguments(compiler: CompilerInfo, optionsArray: string[] | undefined, wasCmake: boolean) {
        if (!optionsArray) optionsArray = [];

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
