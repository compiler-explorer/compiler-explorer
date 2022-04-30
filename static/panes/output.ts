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

import {Toggles} from '../widgets/toggles';
import _ from 'underscore';
import {Pane} from './pane';
import {ga} from '../analytics';
import {Container} from 'golden-layout';
import {PaneState} from './pane.interfaces';
import {Hub} from '../hub';
import * as AnsiToHtml from '../ansi-to-html';
import {OutputState} from './output.interfaces';
import {FontScale} from '../widgets/fontscale';

function makeAnsiToHtml(color?) {
    return new AnsiToHtml.Filter({
        fg: color ? color : '#333',
        bg: '#f5f5f5',
        stream: true,
        escapeXML: true,
    });
}

export class Output extends Pane<OutputState> {
    hub: Hub;
    contentRoot: JQuery<HTMLElement>;
    optionsToolbar: JQuery<HTMLElement>;
    fontScale: FontScale;
    wrapButton: JQuery<HTMLElement>;
    normalAnsiToHtml: AnsiToHtml.Filter;
    errorAnsiToHtml: AnsiToHtml.Filter;
    wrapTitle: string;
    options: Toggles;
    constructor(hub: Hub, container: Container, state: OutputState & PaneState) {
        // canonicalize state
        if ((state as any).compiler) state.id = (state as any).compiler;
        if ((state as any).editor) state.editorid = (state as any).editor;
        if ((state as any).tree) state.treeid = (state as any).tree;
        super(hub, container, state);
        this.hub = hub;
        this.contentRoot = this.domRoot.find('.content');
        this.optionsToolbar = this.domRoot.find('.options-toolbar');
        this.fontScale = new FontScale(this.domRoot, state, '.content');
        this.fontScale.on('change', this.updateState.bind(this));
        this.normalAnsiToHtml = makeAnsiToHtml();
        this.errorAnsiToHtml = makeAnsiToHtml('red');
        this.eventHub.emit('outputOpened', this.compilerInfo.compilerId);
        this.onOptionsChange();
    }

    override getInitialHTML(): string {
        return $('#compiler-output').html();
    }

    override registerOpeningAnalyticsEvent() {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Output',
        });
    }

    override registerButtons(state: OutputState & PaneState) {
        this.wrapButton = this.domRoot.find('.wrap-lines');
        this.wrapTitle = this.wrapButton.prop('title');
        // TODO: Would be nice to be able to get rid of this cast
        this.options = new Toggles(this.domRoot.find('.options'), state as unknown as Record<string, boolean>);
    }

    override registerCallbacks() {
        this.options.on('change', this.onOptionsChange.bind(this));
        this.eventHub.on('compiling', this.onCompiling, this);
    }

    onOptionsChange() {
        const options = this.getEffectiveOptions();
        this.contentRoot.toggleClass('wrap', options.wrap);
        this.wrapButton.prop('title', '[' + (options.wrap ? 'ON' : 'OFF') + '] ' + this.wrapTitle);
        this.updateState();
    }

    getEffectiveOptions() {
        return this.options.get();
    }

    override resize() {
        const rootHeight = this.domRoot.height();
        const toolbarHeight = this.optionsToolbar.height();
        if (rootHeight && toolbarHeight) {
            this.contentRoot.height(rootHeight - toolbarHeight - 5);
        }
    }

    override getCurrentState() {
        const parent = super.getCurrentState();
        const options = this.getEffectiveOptions();
        const state = {
            wrap: options.wrap,
            ...parent,
        };
        this.fontScale.addState(state);
        return state as any;
    }

    addOutputLines(result) {
        const stdout = result.stdout || [];
        const stderr = result.stderr || [];
        for (const obj of stdout.concat(stderr)) {
            const lineNumber = obj.tag ? obj.tag.line : obj.line;
            const columnNumber = obj.tag ? obj.tag.column : -1;
            if (obj.text === '') {
                this.add('<br/>');
            } else {
                this.add(
                    this.normalAnsiToHtml.toHtml(obj.text),
                    lineNumber,
                    columnNumber,
                    obj.tag ? obj.tag.file : false
                );
            }
        }
    }

    onCompiling(compilerId: number) {
        if (this.compilerInfo.compilerId === compilerId) {
            this.setCompileStatus(true);
        }
    }

    override onCompileResult(compilerId: number, compiler: any, result: any) {
        if (compilerId !== this.compilerInfo.compilerId) return;
        if (compiler) this.compilerInfo.compilerName = compiler.name;

        this.contentRoot.empty();

        if (result.buildsteps) {
            for (const step of result.buildsteps) {
                this.add('Step ' + step.step + ' returned: ' + step.code);
                this.addOutputLines(step);
            }
        } else {
            this.addOutputLines(result);
            if (!result.execResult) {
                this.add('Compiler returned: ' + result.code);
            } else {
                this.add('ASM generation compiler returned: ' + result.code);
                this.addOutputLines(result.execResult.buildResult);
                this.add('Execution build compiler returned: ' + result.execResult.buildResult.code);
            }
        }

        if (result.execResult && (result.execResult.didExecute || result.didExecute)) {
            this.add('Program returned: ' + result.execResult.code);
            if (result.execResult.stderr.length || result.execResult.stdout.length) {
                for (const obj of result.execResult.stderr) {
                    // Conserve empty lines as they are discarded by ansiToHtml
                    if (obj.text === '') {
                        this.programOutput('<br/>');
                    } else {
                        this.programOutput(this.errorAnsiToHtml.toHtml(obj.text), 'red');
                    }
                }

                for (const obj of result.execResult.stdout) {
                    // Conserve empty lines as they are discarded by ansiToHtml
                    if (obj.text === '') {
                        this.programOutput('<br/>');
                    } else {
                        this.programOutput(this.normalAnsiToHtml.toHtml(obj.text));
                    }
                }
            }
        }
        this.setCompileStatus(false);
        this.updateTitle();
    }

    override onCompiler(compilerId: number, compiler: unknown, options: unknown, editorId: number, treeId: number) {}

    programOutput(msg: string, color?: string) {
        const elem = $('<div/>').appendTo(this.contentRoot).html(msg).addClass('program-exec-output');

        if (color) elem.css('color', color);
    }

    getEditorIdByFilename(filename) {
        if (this.compilerInfo.treeId) {
            const tree = this.hub.getTreeById(this.compilerInfo.treeId);
            if (tree) {
                return tree.multifileService.getEditorIdByFilename(filename);
            }
            return false;
        }
    }

    emitEditorLinkLine(lineNum, column, filename, goto) {
        if (this.compilerInfo.editorId) {
            this.eventHub.emit('editorLinkLine', this.compilerInfo.editorId, lineNum, column, column + 1, goto);
        } else if (filename) {
            const editorId = this.getEditorIdByFilename(filename);
            if (editorId) {
                this.eventHub.emit('editorLinkLine', editorId, lineNum, column, column + 1, goto);
            }
        }
    }

    add(msg: string, lineNum?: number, column?: number, filename?: string) {
        const elem = $('<div/>').appendTo(this.contentRoot);
        if (lineNum) {
            elem.html(
                $('<span class="linked-compiler-output-line"></span>')
                    .html(msg)
                    .on('click', e => {
                        this.emitEditorLinkLine(lineNum, column, filename, true);
                        // do not bring user to the top of index.html
                        // http://stackoverflow.com/questions/3252730
                        e.preventDefault();
                        return false;
                    })
                    .on('mouseover', () => {
                        this.emitEditorLinkLine(lineNum, column, filename, false);
                    }) as any // TODO
            );
        } else {
            elem.html(msg);
        }
    }

    override getDefaultPaneName() {
        return `Output of ${this.compilerInfo.compilerName}`;
    }

    override getPaneTag() {
        return `(Compiler #${this.compilerInfo.compilerId})`;
    }

    override onCompilerClose(id) {
        if (id === this.compilerInfo.compilerId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            this.close();
            _.defer(() => {
                this.container.close();
            });
        }
    }

    override close() {
        this.eventHub.emit('outputClosed', this.compilerInfo.compilerId);
        this.eventHub.unsubscribe();
    }

    setCompileStatus(isCompiling) {
        this.contentRoot.toggleClass('compiling', isCompiling);
    }
}
