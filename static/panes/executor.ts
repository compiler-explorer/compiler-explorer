// Copyright (c) 2019, Compiler Explorer Authors
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
import $ from 'jquery';
import {ga} from '../analytics.js';
import {Toggles} from '../widgets/toggles.js';
import {FontScale} from '../widgets/fontscale.js';
import {options} from '../options.js';
import {Alert} from '../widgets/alert.js';
import {LibsWidget} from '../widgets/libs-widget.js';
import {Filter as AnsiToHtml} from '../ansi-to-html.js';
import * as TimingWidget from '../widgets/timing-info-widget.js';
import {Settings, SiteSettings} from '../settings.js';
import * as utils from '../utils.js';
import * as LibUtils from '../lib-utils.js';
import {PaneRenaming} from '../widgets/pane-renaming.js';
import {CompilerService} from '../compiler-service.js';
import {Pane} from './pane.js';
import {Hub} from '../hub.js';
import {Container} from 'golden-layout';
import {PaneState} from './pane.interfaces.js';
import {ExecutorState} from './executor.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {Language} from '../../types/languages.interfaces.js';
import {LanguageLibs} from '../options.interfaces.js';
import {
    BypassCache,
    CompilationRequest,
    CompilationRequestOptions,
    CompilationResult,
    FiledataPair,
} from '../../types/compilation/compilation.interfaces.js';
import {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {CompilationStatus as CompilerServiceCompilationStatus} from '../compiler-service.interfaces.js';
import {CompilerPicker} from '../widgets/compiler-picker.js';
import {SourceAndFiles} from '../download-service.js';
import {ICompilerShared} from '../compiler-shared.interfaces.js';
import {CompilerShared} from '../compiler-shared.js';
import {LangInfo} from './compiler-request.interfaces.js';
import {escapeHTML} from '../../shared/common-utils.js';
import {CompilerVersionInfo, setCompilerVersionPopoverForPane} from '../widgets/compiler-version-info.js';

const languages = options.languages;

type CompilationStatus = Omit<CompilerServiceCompilationStatus, 'compilerOut'> & {
    didExecute?: boolean;
};

function makeAnsiToHtml(color?: string): AnsiToHtml {
    return new AnsiToHtml({
        fg: color ? color : '#333',
        bg: '#f5f5f5',
        stream: true,
        escapeXML: true,
    });
}

export class Executor extends Pane<ExecutorState> {
    private contentRoot: JQuery<HTMLElement>;
    private readonly sourceEditorId: number | null;
    private sourceTreeId: number | null;
    private readonly id: number;
    private deferCompiles: boolean;
    private needsCompile: boolean;
    private executionArguments: string;
    private executionStdin: string;
    private source: string;
    private lastTimeTaken: number;
    private pendingRequestSentAt: number;
    private pendingCMakeRequestSentAt: number;
    private nextRequest: CompilationRequest | null;
    private nextCMakeRequest: CompilationRequest | null;
    private options: string;
    private lastResult: CompilationResult | null;
    private alertSystem: Alert;
    private readonly normalAnsiToHtml: AnsiToHtml;
    private readonly errorAnsiToHtml: AnsiToHtml;
    private fontScale: FontScale;
    private compilerPicker: CompilerPicker;
    private currentLangId: string;
    private toggleWrapButton: Toggles;
    private outputContentRoot: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private executionStatusSection: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private compilerOutputSection: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private executionOutputSection: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private optionsField: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private execArgsField: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private execStdinField: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private prependOptions: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private fullCompilerName: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private fullTimingInfo: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private libsButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private compileTimeLabel: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private shortCompilerName: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private bottomBar: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private statusLabel: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private statusIcon: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]> | null;
    private panelCompilation: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private panelArgs: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private panelStdin: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private wrapTitle: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private rerunButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private compileClearCache: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private wrapButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private toggleCompilation: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private toggleArgs: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private toggleStdin: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private toggleCompilerOut: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private libsWidget?: LibsWidget;
    private readonly infoByLang: Record<string, LangInfo | undefined>;
    private compiler: CompilerInfo | null;
    private compilerShared: ICompilerShared;

    constructor(hub: Hub, container: Container, state: PaneState & ExecutorState) {
        super(hub, container, state);
        if (this.sourceTreeId) {
            this.sourceEditorId = null;
        } else {
            this.sourceEditorId = state.source || 1;
        }
        this.id = state.id || this.hub.nextExecutorId();

        this.contentRoot = this.domRoot.find('.content');
        this.infoByLang = {};
        this.deferCompiles = true;
        this.needsCompile = false;
        this.source = '';
        this.lastResult = {code: -1, timedOut: false, stdout: [], stderr: []};
        this.lastTimeTaken = 0;
        this.pendingRequestSentAt = 0;
        this.pendingCMakeRequestSentAt = 0;
        this.nextRequest = null;
        this.nextCMakeRequest = null;

        this.alertSystem = new Alert();
        this.alertSystem.prefixMessage = 'Executor #' + this.id;

        this.normalAnsiToHtml = makeAnsiToHtml();
        this.errorAnsiToHtml = makeAnsiToHtml('red');

        this.initButtons(state);

        this.fontScale = new FontScale(this.domRoot, state, 'pre.content');
        this.compilerPicker = new CompilerPicker(
            this.domRoot,
            this.hub,
            this.currentLangId,
            this.compiler ? this.compiler.id : '',
            this.onCompilerChange.bind(this),
            this.compilerIsVisible,
        );

        this.initLibraries(state);
        this.compilerShared = new CompilerShared(this.domRoot, this.onCompilerOverridesChange.bind(this));
        this.compilerShared.updateState(state);
        this.initCallbacks();
        // Handle initial settings
        this.onSettingsChange(this.settings);
        this.updateCompilerInfo();
        this.updateState();

        if (this.sourceTreeId) {
            this.compile();
        }

        if (!this.hub.deferred) {
            this.undefer();
        }
    }

    override initializeStateDependentProperties(state: PaneState & ExecutorState) {
        this.sourceTreeId = state.tree ?? null;
        this.settings = Settings.getStoredSettings();
        this.initLangAndCompiler(state);
        this.options = state.options || options.compileOptions[this.currentLangId];
        this.executionArguments = state.execArgs || '';
        this.executionStdin = state.execStdin || '';
        this.paneRenaming = new PaneRenaming(this, state);
    }

    override getInitialHTML(): string {
        return $('#executor').html();
    }

    compilerIsVisible(compiler: CompilerInfo): boolean {
        return !!compiler.supportsExecute;
    }

    getEditorIdByFilename(filename: string): number | null {
        if (this.sourceTreeId) {
            const tree = this.hub.getTreeById(this.sourceTreeId);
            if (tree) {
                return tree.multifileService.getEditorIdByFilename(filename);
            }
        } else if (this.sourceEditorId) {
            return this.sourceEditorId;
        }
        return null;
    }

    initLangAndCompiler(state: PaneState & ExecutorState): void {
        const langId = state.lang ?? null;
        const compilerId = state.compiler;
        const result = this.hub.compilerService.processFromLangAndCompiler(langId, compilerId);
        this.compiler = result?.compiler ?? null;
        this.currentLangId = result?.langId ?? '';
        this.updateLibraries();
    }

    close(): void {
        this.eventHub.unsubscribe();
        if (this.compilerPicker instanceof CompilerPicker) {
            this.compilerPicker.destroy();
        }

        this.eventHub.emit('executorClose', this.id);
    }

    undefer(): void {
        this.deferCompiles = false;
        if (this.needsCompile) this.compile();
    }

    override resize(): void {
        _.defer(self => {
            let topBarHeight = utils.updateAndCalcTopBarHeight(self.domRoot, $(self.topBar[0]), self.hideable);

            // We have some more elements that modify the topBarHeight
            if (!self.panelCompilation.hasClass('d-none')) {
                topBarHeight += self.panelCompilation.outerHeight(true);
            }
            if (!self.panelArgs.hasClass('d-none')) {
                topBarHeight += self.panelArgs.outerHeight(true);
            }
            if (!self.panelStdin.hasClass('d-none')) {
                topBarHeight += self.panelStdin.outerHeight(true);
            }

            const bottomBarHeight = self.bottomBar.outerHeight(true);
            self.outputContentRoot.outerHeight(self.domRoot.height() - topBarHeight - bottomBarHeight);
        }, this);
    }

    private errorResult(message: string): CompilationResult {
        return {stdout: [], timedOut: false, code: -1, stderr: [{text: message}]};
    }

    compile(bypassCache?: BypassCache): void {
        if (this.deferCompiles) {
            this.needsCompile = true;
            return;
        }
        this.needsCompile = false;
        this.compileTimeLabel.text(' - Compiling...');
        const options: CompilationRequestOptions = {
            userArguments: this.options,
            executeParameters: {
                args: this.executionArguments,
                stdin: this.executionStdin,
            },
            compilerOptions: {
                executorRequest: true,
                skipAsm: true,
                overrides: this.compilerShared.getOverrides(),
            },
            filters: {execute: true},
            tools: [],
            libraries: [],
        };

        this.libsWidget?.getLibsInUse()?.forEach(item => {
            options.libraries.push({
                id: item.libId,
                version: item.versionId,
            });
        });

        if (this.sourceTreeId) {
            this.compileFromTree(options, bypassCache);
        } else {
            this.compileFromEditorSource(options, bypassCache);
        }
    }

    compileFromEditorSource(options: CompilationRequestOptions, bypassCache?: BypassCache): void {
        if (!this.compiler?.supportsExecute) {
            this.alertSystem.notify('This compiler (' + this.compiler?.name + ') does not support execution', {
                group: 'execution',
            });
            return;
        }
        this.hub.compilerService.expandToFiles(this.source).then((sourceAndFiles: SourceAndFiles) => {
            const request: CompilationRequest = {
                source: sourceAndFiles.source || '',
                compiler: this.compiler ? this.compiler.id : '',
                options: options,
                lang: this.currentLangId,
                files: sourceAndFiles.files,
            };
            if (bypassCache) request.bypassCache = bypassCache;
            if (!this.compiler) {
                this.onCompileResponse(request, this.errorResult('<Please select a compiler>'), false);
            } else {
                this.sendCompile(request);
            }
        });
    }

    compileFromTree(options: CompilationRequestOptions, bypassCache?: BypassCache): void {
        const tree = this.hub.getTreeById(this.sourceTreeId ?? -1);
        if (!tree) {
            this.sourceTreeId = null;
            this.compileFromEditorSource(options, bypassCache);
            return;
        }

        const request: CompilationRequest = {
            source: tree.multifileService.getMainSource(),
            compiler: this.compiler ? this.compiler.id : '',
            options: options,
            lang: this.currentLangId,
            files: tree.multifileService.getFiles(),
        };

        const fetches: Promise<void>[] = [];
        fetches.push(
            this.hub.compilerService.expandToFiles(request.source).then((sourceAndFiles: SourceAndFiles) => {
                request.source = sourceAndFiles.source;
                request.files.push(...sourceAndFiles.files);
            }),
        );

        const moreFiles: FiledataPair[] = [];
        for (let i = 0; i < request.files.length; i++) {
            const file = request.files[i];
            fetches.push(
                this.hub.compilerService.expandToFiles(file.contents).then((sourceAndFiles: SourceAndFiles) => {
                    file.contents = sourceAndFiles.source;
                    moreFiles.push(...sourceAndFiles.files);
                }),
            );
        }
        request.files.push(...moreFiles);

        Promise.all(fetches).then(() => {
            const treeState = tree.currentState();
            const cmakeProject = tree.multifileService.isACMakeProject();

            if (bypassCache) request.bypassCache = bypassCache;
            if (!this.compiler) {
                this.onCompileResponse(request, this.errorResult('<Please select a compiler>'), false);
            } else if (cmakeProject && request.source === '') {
                this.onCompileResponse(request, this.errorResult('<Please supply a CMakeLists.txt>'), false);
            } else {
                if (cmakeProject) {
                    request.options.compilerOptions.cmakeArgs = treeState.cmakeArgs;
                    request.options.compilerOptions.customOutputFilename = treeState.customOutputFilename;
                    this.sendCMakeCompile(request);
                } else {
                    this.sendCompile(request);
                }
            }
        });
    }

    sendCMakeCompile(request: CompilationRequest): void {
        const onCompilerResponse = this.onCMakeResponse.bind(this);

        if (this.pendingCMakeRequestSentAt) {
            // If we have a request pending, then just store this request to do once the
            // previous request completes.
            this.nextCMakeRequest = request;
            return;
        }
        // this.eventHub.emit('compiling', this.id, this.compiler);
        // Display the spinner
        this.handleCompilationStatus({code: 4});
        this.pendingCMakeRequestSentAt = Date.now();
        // After a short delay, give the user some indication that we're working on their
        // compilation.
        this.hub.compilerService
            .submitCMake(request)
            .then((x: any) => {
                onCompilerResponse(request, x.result, x.localCacheHit);
            })
            .catch(x => {
                let message = 'Unknown error';
                if (_.isString(x)) {
                    message = x;
                } else if (x) {
                    message = x.error || x.code || x.message || x;
                }
                onCompilerResponse(request, this.errorResult(message), false);
            });
    }

    sendCompile(request: CompilationRequest): void {
        const onCompilerResponse = this.onCompileResponse.bind(this);

        if (this.pendingRequestSentAt) {
            // If we have a request pending, then just store this request to do once the
            // previous request completes.
            this.nextRequest = request;
            return;
        }
        // this.eventHub.emit('compiling', this.id, this.compiler);
        // Display the spinner
        this.handleCompilationStatus({code: 4});
        this.pendingRequestSentAt = Date.now();
        // After a short delay, give the user some indication that we're working on their
        // compilation.
        this.hub.compilerService
            .submit(request)
            .then((x: any) => {
                onCompilerResponse(request, x.result, x.localCacheHit);
            })
            .catch(x => {
                let message = 'Unknown error';
                if (typeof x === 'string') {
                    message = x;
                } else if (x) {
                    message = x.error || x.code || x.message || x;
                }
                onCompilerResponse(request, this.errorResult(message), false);
            });
    }

    addCompilerOutputLine(
        msg: string,
        container: JQuery,
        lineNum: number | undefined,
        column: number | undefined,
        addLineLinks: boolean,
        filename: string | null,
    ): void {
        const elem = $('<div/>').appendTo(container);
        if (addLineLinks && lineNum) {
            elem.empty();
            $('<span class="linked-compiler-output-line"></span>')
                .html(msg)
                .on('click', e => {
                    const editorId = this.getEditorIdByFilename(filename ?? '');
                    if (editorId) {
                        this.eventHub.emit('editorLinkLine', editorId, lineNum, column ?? 0, (column ?? 0) + 1, true);
                    }
                    // do not bring user to the top of index.html
                    // http://stackoverflow.com/questions/3252730
                    e.preventDefault();
                    return false;
                })
                .on('mouseover', () => {
                    const editorId = this.getEditorIdByFilename(filename ?? '');
                    if (editorId) {
                        this.eventHub.emit('editorLinkLine', editorId, lineNum, column ?? 0, (column ?? 0) + 1, false);
                    }
                })
                .appendTo(elem);
        } else {
            elem.html(msg);
        }
    }

    clearPreviousOutput(): void {
        this.executionStatusSection.empty();
        this.compilerOutputSection.empty();
        this.executionOutputSection.empty();
    }

    handleOutput(
        output: ResultLine[],
        element: JQuery<HTMLElement>,
        ansiParser: AnsiToHtml,
        addLineLinks: boolean,
    ): JQuery<HTMLElement> {
        const outElem = $('<pre class="card"></pre>').appendTo(element);
        output.forEach(obj => {
            if (obj.text === '') {
                this.addCompilerOutputLine('<br/>', outElem, undefined, undefined, false, null);
            } else {
                const lineNumber = obj.tag ? obj.tag.line : obj.line;
                const columnNumber = obj.tag ? obj.tag.column : -1;
                const filename = obj.tag ? obj.tag.file : false;
                this.addCompilerOutputLine(
                    ansiParser.toHtml(obj.text),
                    outElem,
                    lineNumber,
                    columnNumber,
                    addLineLinks,
                    filename || null,
                );
            }
        });
        return outElem;
    }

    getBuildStdoutFromResult(result: CompilationResult): ResultLine[] {
        let arr: ResultLine[] = [];

        if (result.buildResult) {
            arr = arr.concat(result.buildResult.stdout);
        }

        if (result.buildsteps) {
            result.buildsteps.forEach(step => {
                arr = arr.concat(step.stdout);
            });
        }

        return arr;
    }

    getBuildStderrFromResult(result: CompilationResult): ResultLine[] {
        let arr: ResultLine[] = [];

        if (result.buildResult) {
            arr = arr.concat(result.buildResult.stderr);
        }

        if (result.buildsteps) {
            result.buildsteps.forEach(step => {
                arr = arr.concat(step.stderr);
            });
        }

        return arr;
    }

    getExecutionStdoutfromResult(result: CompilationResult): ResultLine[] {
        if (result.execResult && result.execResult.stdout !== undefined) {
            return result.execResult.stdout;
        }

        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        return result.stdout || [];
    }

    getExecutionStderrfromResult(result: CompilationResult): ResultLine[] {
        if (result.execResult) {
            return result.execResult.stderr as ResultLine[];
        }

        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        return result.stderr || [];
    }

    onCMakeResponse(request: CompilationRequest, result: CompilationResult, cached: boolean): void {
        result.source = this.source;
        this.lastResult = result;
        const timeTaken = Math.max(0, Date.now() - this.pendingCMakeRequestSentAt);
        this.lastTimeTaken = timeTaken;
        const wasRealReply = this.pendingCMakeRequestSentAt > 0;
        this.pendingCMakeRequestSentAt = 0;

        this.handleCompileRequestAndResponse(request, result, cached, wasRealReply, timeTaken);

        this.doNextCMakeRequest();
    }

    doNextCompileRequest(): void {
        if (this.nextRequest) {
            const next = this.nextRequest;
            this.nextRequest = null;
            this.sendCompile(next);
        }
    }

    doNextCMakeRequest(): void {
        if (this.nextCMakeRequest) {
            const next = this.nextCMakeRequest;
            this.nextCMakeRequest = null;
            this.sendCMakeCompile(next);
        }
    }

    handleCompileRequestAndResponse(
        request: CompilationRequest,
        result: CompilationResult,
        cached: boolean,
        wasRealReply: boolean,
        timeTaken: number,
    ): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'Compile',
            eventAction: request.compiler,
            eventLabel: request.options.userArguments,
            eventValue: cached ? 1 : 0,
        });
        ga.proxy('send', {
            hitType: 'timing',
            timingCategory: 'Compile',
            timingVar: request.compiler,
            timingValue: timeTaken,
        });

        this.clearPreviousOutput();
        const compileStdout = this.getBuildStdoutFromResult(result);
        const compileStderr = this.getBuildStderrFromResult(result);
        const execStdout = this.getExecutionStdoutfromResult(result);
        const execStderr = this.getExecutionStderrfromResult(result);

        let buildResultCode = 0;

        if (result.buildResult) {
            buildResultCode = result.buildResult.code;
        } else if (result.buildsteps) {
            result.buildsteps.forEach(step => {
                buildResultCode = step.code;
            });
        }

        if (!result.didExecute) {
            this.executionStatusSection.append($('<div/>').text('Could not execute the program'));
            this.executionStatusSection.append($('<div/>').text('Compiler returned: ' + buildResultCode));
        }
        // reset stream styles
        this.normalAnsiToHtml.reset();
        this.errorAnsiToHtml.reset();
        if (compileStdout.length > 0) {
            this.compilerOutputSection.append($('<div/>').text('Compiler stdout'));
            this.handleOutput(compileStdout, this.compilerOutputSection, this.normalAnsiToHtml, true);
        }
        if (compileStderr.length > 0) {
            this.compilerOutputSection.append($('<div/>').text('Compiler stderr'));
            this.handleOutput(compileStderr, this.compilerOutputSection, this.errorAnsiToHtml, true);
        }
        if (result.didExecute) {
            const exitCode = result.execResult ? result.execResult.code : result.code;
            this.executionOutputSection.append($('<div/>').text('Program returned: ' + exitCode));
            if (execStdout.length > 0) {
                this.executionOutputSection.append($('<div/>').text('Program stdout'));
                const outElem = this.handleOutput(
                    execStdout,
                    this.executionOutputSection,
                    this.normalAnsiToHtml,
                    false,
                );
                outElem.addClass('execution-stdout');
            }
            if (execStderr.length > 0) {
                this.executionOutputSection.append($('<div/>').text('Program stderr'));
                this.handleOutput(execStderr, this.executionOutputSection, this.normalAnsiToHtml, false);
            }
        }

        this.handleCompilationStatus({code: 1, didExecute: result.didExecute});
        let timeLabelText = '';
        if (cached) {
            timeLabelText = ' - cached';
        } else if (wasRealReply) {
            timeLabelText = ' - ' + timeTaken + 'ms';
        }
        this.compileTimeLabel.text(timeLabelText);

        this.setCompilationOptionsPopover(result.buildResult ? result.buildResult.compilationOptions.join(' ') : '');

        if (this.currentLangId)
            this.eventHub.emit('executeResult', this.id, this.compiler, result, languages[this.currentLangId]);
    }

    onCompileResponse(request: CompilationRequest, result: CompilationResult, cached: boolean): void {
        // Save which source produced this change. It should probably be saved earlier though
        result.source = this.source;
        this.lastResult = result;
        const timeTaken = Math.max(0, Date.now() - this.pendingRequestSentAt);
        this.lastTimeTaken = timeTaken;
        const wasRealReply = this.pendingRequestSentAt > 0;
        this.pendingRequestSentAt = 0;

        this.handleCompileRequestAndResponse(request, result, cached, wasRealReply, timeTaken);

        this.doNextCompileRequest();
    }

    resendResult(): boolean {
        if (!$.isEmptyObject(this.lastResult)) {
            this.eventHub.emit('executeResult', this.id, this.compiler, this.lastResult, languages[this.currentLangId]);
            return true;
        }
        return false;
    }

    onResendExecutionResult(id: number): void {
        if (id === this.id) {
            this.resendResult();
        }
    }

    onEditorChange(editor: number, source: string, langId: string, compilerId?: number): void {
        if (this.sourceTreeId) {
            const tree = this.hub.getTreeById(this.sourceTreeId);
            if (tree) {
                if (tree.multifileService.isEditorPartOfProject(editor)) {
                    if (this.settings.compileOnChange) {
                        this.compile();

                        return;
                    }
                }
            }
        }

        if (editor === this.sourceEditorId && langId === this.currentLangId && compilerId === undefined) {
            this.source = source;
            if (this.settings.compileOnChange) {
                this.compile();
            }
        }
    }

    initButtons(state: PaneState & ExecutorState): void {
        this.outputContentRoot = this.domRoot.find('pre.content');
        this.executionStatusSection = this.outputContentRoot.find('.execution-status');
        this.compilerOutputSection = this.outputContentRoot.find('.compiler-output');
        this.executionOutputSection = this.outputContentRoot.find('.execution-output');
        this.toggleWrapButton = new Toggles(this.domRoot.find('.options'), state as unknown as Record<string, boolean>);

        this.optionsField = this.domRoot.find('.compilation-options');
        this.execArgsField = this.domRoot.find('.execution-arguments');
        this.execStdinField = this.domRoot.find('.execution-stdin');
        this.prependOptions = this.domRoot.find('.prepend-options');
        this.fullCompilerName = this.domRoot.find('.full-compiler-name');
        this.fullTimingInfo = this.domRoot.find('.full-timing-info');
        this.setCompilationOptionsPopover(this.compiler?.options ?? null);

        this.compileTimeLabel = this.domRoot.find('.compile-time');
        this.libsButton = this.domRoot.find('.btn.show-libs');

        // Dismiss on any click that isn't either in the opening element, inside
        // the popover or on any alert
        $(document).on('mouseup', e => {
            const target = $(e.target);
            if (
                !target.is(this.prependOptions) &&
                this.prependOptions.has(target as any).length === 0 &&
                target.closest('.popover').length === 0
            )
                this.prependOptions.popover('hide');

            if (
                !target.is(this.fullCompilerName) &&
                this.fullCompilerName.has(target as any).length === 0 &&
                target.closest('.popover').length === 0
            )
                this.fullCompilerName.popover('hide');
        });

        this.optionsField.val(this.options);
        this.execArgsField.val(this.executionArguments);
        this.execStdinField.val(this.executionStdin);

        this.shortCompilerName = this.domRoot.find('.short-compiler-name');
        this.setCompilerVersionPopover();

        this.topBar = this.domRoot.find('.top-bar');
        this.bottomBar = this.domRoot.find('.bottom-bar');
        this.statusLabel = this.domRoot.find('.status-text');

        this.hideable = this.domRoot.find('.hideable');
        this.statusIcon = this.domRoot.find('.status-icon');

        this.panelCompilation = this.domRoot.find('.panel-compilation');
        this.panelArgs = this.domRoot.find('.panel-args');
        this.panelStdin = this.domRoot.find('.panel-stdin');

        this.wrapButton = this.domRoot.find('.wrap-lines');
        this.wrapTitle = this.wrapButton.prop('title');

        this.rerunButton = this.bottomBar.find('.rerun');
        this.compileClearCache = this.bottomBar.find('.clear-cache');

        this.initToggleButtons(state);
    }

    initToggleButtons(state: PaneState & ExecutorState): void {
        this.toggleCompilation = this.domRoot.find('.toggle-compilation');
        this.toggleArgs = this.domRoot.find('.toggle-args');
        this.toggleStdin = this.domRoot.find('.toggle-stdin');
        this.toggleCompilerOut = this.domRoot.find('.toggle-compilerout');

        if (!state.compilationPanelShown) {
            this.hidePanel(this.toggleCompilation, this.panelCompilation);
        }

        if (state.argsPanelShown) {
            this.showPanel(this.toggleArgs, this.panelArgs);
        }

        if (state.stdinPanelShown) {
            this.showPanel(this.toggleStdin, this.panelStdin);
        }

        if (!state.compilerOutShown) {
            this.hidePanel(this.toggleCompilerOut, this.compilerOutputSection);
        }

        if (state.wrap === true) {
            this.contentRoot.addClass('wrap');
            this.wrapButton.prop('title', '[ON] ' + this.wrapTitle);
        } else {
            this.contentRoot.removeClass('wrap');
            this.wrapButton.prop('title', '[OFF] ' + this.wrapTitle);
        }
    }

    onLibsChanged(): void {
        this.updateState();
        this.compile();
    }

    initLibraries(state: PaneState & ExecutorState): void {
        this.libsWidget = new LibsWidget(
            this.currentLangId,
            this.compiler,
            this.libsButton,
            state,
            this.onLibsChanged.bind(this),
            LibUtils.getSupportedLibraries(
                this.compiler ? this.compiler.libsArr : [],
                this.currentLangId,
                this.compiler?.remote ?? null,
            ),
        );
    }

    onFontScale(): void {
        this.updateState();
    }

    initListeners(): void {
        // this.filters.on('change', this.onFilterChange.bind(this));
        this.fontScale.on('change', this.onFontScale.bind(this));
        this.paneRenaming.on('renamePane', this.updateState.bind(this));
        this.toggleWrapButton.on('change', this.onToggleWrapChange.bind(this));

        this.container.on('destroy', this.close, this);
        this.container.on('resize', this.resize, this);
        this.container.on('shown', this.resize, this);
        this.container.on('open', () => {
            this.eventHub.emit('executorOpen', this.id, this.sourceEditorId ?? false);
        });
        this.eventHub.on('editorChange', this.onEditorChange, this);
        this.eventHub.on('editorClose', this.onEditorClose, this);
        this.eventHub.on('settingsChange', this.onSettingsChange, this);
        this.eventHub.on('requestCompilation', this.onRequestCompilation, this);
        this.eventHub.on('resendExecution', this.onResendExecutionResult, this);
        this.eventHub.on('resize', this.resize, this);
        this.eventHub.on('findExecutors', this.sendExecutor, this);
        this.eventHub.on('languageChange', this.onLanguageChange, this);

        this.fullTimingInfo.off('click').on('click', () => {
            TimingWidget.displayCompilationTiming(this.lastResult, this.lastTimeTaken);
        });
    }

    showPanel(button: JQuery<HTMLElement>, panel: JQuery<HTMLElement>): void {
        panel.removeClass('d-none');
        button.addClass('active');
        this.resize();
    }

    hidePanel(button: JQuery<HTMLElement>, panel: JQuery<HTMLElement>): void {
        panel.addClass('d-none');
        button.removeClass('active');
        this.resize();
    }

    togglePanel(button: JQuery<HTMLElement>, panel: JQuery<HTMLElement>): void {
        if (panel.hasClass('d-none')) {
            this.showPanel(button, panel);
        } else {
            this.hidePanel(button, panel);
        }
        this.updateState();
    }

    initCallbacks(): void {
        this.initListeners();

        const optionsChange = _.debounce(e => {
            this.onOptionsChange($(e.target).val() as string);
        }, 800);

        const execArgsChange = _.debounce(e => {
            this.onExecArgsChange($(e.target).val() as string);
        }, 800);

        const execStdinChange = _.debounce(e => {
            this.onExecStdinChange($(e.target).val() as string);
        }, 800);

        this.optionsField.on('change', optionsChange).on('keyup', optionsChange);

        this.execArgsField.on('change', execArgsChange).on('keyup', execArgsChange);

        this.execStdinField.on('change', execStdinChange).on('keyup', execStdinChange);

        // Dismiss the popover on escape.
        $(document).on('keyup.editable', e => {
            if (e.which === 27) {
                this.libsButton.popover('hide');
            }
        });

        this.toggleCompilation.on('click', () => {
            this.togglePanel(this.toggleCompilation, this.panelCompilation);
        });

        this.toggleArgs.on('click', () => {
            this.togglePanel(this.toggleArgs, this.panelArgs);
        });

        this.toggleStdin.on('click', () => {
            this.togglePanel(this.toggleStdin, this.panelStdin);
        });

        this.toggleCompilerOut.on('click', () => {
            this.togglePanel(this.toggleCompilerOut, this.compilerOutputSection);
        });

        this.rerunButton.on('click', () => {
            this.compile(BypassCache.Execution);
        });

        this.compileClearCache.on('click', () => {
            this.compile(BypassCache.Compilation);
        });

        // Dismiss on any click that isn't either in the opening element, inside
        // the popover or on any alert
        $(document).on('click', e => {
            const elem = this.libsButton;
            const target = $(e.target);
            if (!target.is(elem) && elem.has(target as any).length === 0 && target.closest('.popover').length === 0) {
                elem.popover('hide');
            }
        });

        this.eventHub.on('initialised', this.undefer, this);

        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (MutationObserver !== undefined) {
            new MutationObserver(this.resize.bind(this)).observe(this.execStdinField[0], {
                attributes: true,
                attributeFilter: ['style'],
            });
        }
    }

    shouldEmitExecutionOnFieldChange(): boolean {
        return this.settings.executorCompileOnChange;
    }

    onOptionsChange(options: string): void {
        this.options = options;
        this.updateState();
        if (this.shouldEmitExecutionOnFieldChange()) {
            this.compile();
        }
    }

    onExecArgsChange(args: string): void {
        this.executionArguments = args;
        this.updateState();
        if (this.shouldEmitExecutionOnFieldChange()) {
            this.compile();
        }
    }

    onCompilerOverridesChange(): void {
        this.updateState();
        if (this.shouldEmitExecutionOnFieldChange()) {
            this.compile();
        }
    }

    onExecStdinChange(newStdin: string): void {
        this.executionStdin = newStdin;
        this.updateState();
        if (this.shouldEmitExecutionOnFieldChange()) {
            this.compile();
        }
    }

    onRequestCompilation(editorId: number | boolean, treeId: number | boolean): void {
        if (editorId === this.sourceEditorId || (treeId && treeId === this.sourceTreeId)) {
            this.compile();
        }
    }

    updateCompilerInfo(): void {
        this.updateCompilerName();
        if (this.compiler) {
            if (this.compiler.notification) {
                this.alertSystem.notify(this.compiler.notification, {
                    group: 'compilerwarning',
                    alertClass: 'notification-info',
                    dismissTime: 5000,
                });
            }
            this.prependOptions.data('content', this.compiler.options);
        }
        this.sendExecutor();
    }

    updateCompilerUI(): void {
        this.updateCompilerInfo();
        // Resize in case the new compiler name is too big
        this.resize();
    }

    onCompilerChange(value: string): void {
        this.compiler = this.hub.compilerService.findCompiler(this.currentLangId, value);
        this.updateLibraries();
        this.updateState();
        this.compile();
        this.updateCompilerUI();
    }

    onToggleWrapChange(): void {
        const state = this.getCurrentState();
        this.contentRoot.toggleClass('wrap', state.wrap);
        this.wrapButton.prop('title', '[' + (state.wrap ? 'ON' : 'OFF') + '] ' + this.wrapTitle);
        this.updateState();
    }

    sendExecutor(): void {
        this.eventHub.emit(
            'executor',
            this.id,
            this.compiler,
            this.options,
            this.sourceEditorId ?? -1,
            this.sourceTreeId ?? -1,
        );
    }

    onEditorClose(editor: number): void {
        if (editor === this.sourceEditorId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            this.close();
            _.defer(function (self) {
                self.container.close();
            }, this);
        }
    }

    override getCurrentState(): ExecutorState & PaneState {
        const state: ExecutorState & PaneState = {
            id: this.id,
            compilerName: '',
            compiler: this.compiler ? this.compiler.id : '',
            source: this.sourceEditorId ?? undefined,
            tree: this.sourceTreeId ?? undefined,
            options: this.options,
            execArgs: this.executionArguments,
            execStdin: this.executionStdin,
            libs: this.libsWidget?.get(),
            lang: this.currentLangId,
            compilationPanelShown: !this.panelCompilation.hasClass('d-none'),
            compilerOutShown: !this.compilerOutputSection.hasClass('d-none'),
            argsPanelShown: !this.panelArgs.hasClass('d-none'),
            stdinPanelShown: !this.panelStdin.hasClass('d-none'),
            wrap: this.toggleWrapButton.get().wrap,
            overrides: this.compilerShared.getOverrides(),
        };

        this.paneRenaming.addState(state);
        this.fontScale.addState(state);
        return state;
    }

    override updateState(): void {
        const state = this.getCurrentState();
        this.container.setState(state);
        this.compilerShared.updateState(state);
    }

    getCompilerName(): string {
        return this.compiler ? this.compiler.name : 'No compiler set';
    }

    getLanguageName(): string {
        const lang = this.currentLangId ? (options.languages[this.currentLangId] as Language | undefined) : undefined;
        return lang ? lang.name : '?';
    }

    getLinkHint(): string {
        if (this.sourceTreeId) {
            return 'Tree #' + this.sourceTreeId;
        } else {
            return 'Editor #' + this.sourceEditorId;
        }
    }

    override getPaneName(): string {
        const langName = this.getLanguageName();
        const compName = this.getCompilerName();
        return 'Executor ' + compName + ' (' + langName + ', ' + this.getLinkHint() + ')';
    }

    override updateTitle(): void {
        const name = this.paneName ? this.paneName : this.getPaneName();
        this.container.setTitle(escapeHTML(name));
    }

    updateCompilerName() {
        this.updateTitle();
        const compilerName = this.getCompilerName();
        const compilerVersion = this.compiler?.version ?? '';
        const compilerFullVersion = this.compiler?.fullVersion ?? compilerVersion;
        const compilerNotification = this.compiler?.notification ?? '';
        this.shortCompilerName.text(compilerName);
        this.setCompilerVersionPopover(
            {
                version: compilerVersion,
                fullVersion: compilerFullVersion,
            },
            compilerNotification,
            this.compiler?.id,
        );
    }

    setCompilationOptionsPopover(content: string | null) {
        this.prependOptions.popover('dispose');
        this.prependOptions.popover({
            content: content || 'No options in use',
            template:
                '<div class="popover' +
                (content ? ' compiler-options-popover' : '') +
                '" role="tooltip"><div class="arrow"></div>' +
                '<h3 class="popover-header"></h3><div class="popover-body"></div></div>',
        });
    }

    setCompilerVersionPopover(version?: CompilerVersionInfo, notification?: string[] | string, compilerId?: string) {
        setCompilerVersionPopoverForPane(this, version, notification, compilerId);
    }

    override onSettingsChange(newSettings: SiteSettings): void {
        this.settings = _.clone(newSettings);
    }

    private ariaLabel(status: CompilationStatus): string {
        // Compiling...
        if (status.code === 4) return 'Compiling';
        if (status.didExecute) {
            return 'Program compiled & executed';
        } else {
            return 'Program could not be executed';
        }
    }

    private color(status: CompilationStatus) {
        // Compiling...
        if (status.code === 4) return '#888888';
        if (status.didExecute) return '#12BB12';
        return '#FF1212';
    }

    // TODO: Duplicate with compiler-service.ts?
    handleCompilationStatus(status: CompilationStatus): void {
        // We want to do some custom styles for the icon, so we don't pass it here and instead do it later
        CompilerService.handleCompilationStatus(this.statusLabel, null, {compilerOut: 0, ...status});

        if (this.statusIcon != null) {
            this.statusIcon
                .removeClass()
                .addClass('status-icon fas')
                .css('color', this.color(status))
                .toggle(status.code !== 0)
                .attr('aria-label', this.ariaLabel(status))
                .toggleClass('fa-spinner fa-spin', status.code === 4)
                .toggleClass('fa-times-circle', status.code !== 4 && !status.didExecute)
                .toggleClass('fa-check-circle', status.code !== 4 && status.didExecute);
        }
    }

    updateLibraries(): void {
        if (this.libsWidget) {
            let filteredLibraries: LanguageLibs = {};
            if (this.compiler) {
                filteredLibraries = LibUtils.getSupportedLibraries(
                    this.compiler.libsArr,
                    this.currentLangId || '',
                    this.compiler.remote ?? null,
                );
            }

            this.libsWidget.setNewLangId(this.currentLangId, this.compiler?.id ?? '', filteredLibraries);
        }
    }

    onLanguageChange(editorId: number | boolean, newLangId: string): void {
        if (this.sourceEditorId === editorId && this.currentLangId) {
            const oldLangId = this.currentLangId;
            this.currentLangId = newLangId;
            // Store the current selected stuff to come back to it later in the same session (Not state stored!)
            this.infoByLang[oldLangId] = {
                compiler: this.compiler && this.compiler.id ? this.compiler.id : options.defaultCompiler[oldLangId],
                options: this.options,
                execArgs: this.executionArguments,
                execStdin: this.executionStdin,
            };
            const info = this.infoByLang[this.currentLangId];
            this.initLangAndCompiler({compilerName: '', id: 0, lang: newLangId, compiler: info?.compiler ?? ''});
            this.updateCompilersSelector(info);
            this.updateCompilerUI();
            this.updateState();
        }
    }

    getCurrentLangCompilers(): CompilerInfo[] {
        const allCompilers: Record<string, CompilerInfo> | undefined = this.hub.compilerService.getCompilersForLang(
            this.currentLangId,
        );
        if (!allCompilers) return [];

        const hasAtLeastOneExecuteSupported = Object.values(allCompilers).some(compiler => {
            return compiler.supportsExecute !== false;
        });

        if (!hasAtLeastOneExecuteSupported) {
            this.compiler = null;
            return [];
        }

        return Object.values(allCompilers).filter(compiler => {
            return (
                (compiler.hidden !== true && compiler.supportsExecute !== false) ||
                (this.compiler && compiler.id === this.compiler.id)
            );
        });
    }

    updateCompilersSelector(info: LangInfo | undefined): void {
        this.compilerPicker.update(this.currentLangId, this.compiler?.id ?? '');
        this.options = info?.options || '';
        this.optionsField.val(this.options);
        this.executionArguments = info?.execArgs || '';
        this.execArgsField.val(this.executionArguments);
        this.executionStdin = info?.execStdin || '';
        this.execStdinField.val(this.executionStdin);
    }

    getDefaultPaneName(): string {
        return '';
    }

    onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void {}

    onCompiler(compilerId: number, compiler: CompilerInfo, options: string, editorId: number, treeId: number): void {}

    registerOpeningAnalyticsEvent(): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Executor',
        });
    }
}
