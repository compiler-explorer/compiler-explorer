// Copyright (c) 2025, Compiler Explorer Authors
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

import {Container} from 'golden-layout';
import $ from 'jquery';
import {marked} from 'marked';
import _ from 'underscore';
// No longer needed with flexbox layout
import {CompilationResult} from '../compilation/compilation.interfaces.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {Hub} from '../hub.js';
import {options} from '../options.js';
import {SentryCapture} from '../sentry.js';
import * as utils from '../utils.js';
import {FontScale} from '../widgets/fontscale.js';
import {PaneState} from './pane.interfaces.js';
import {Pane} from './pane.js';

interface ClaudeExplainResponse {
    status: string;
    explanation: string;
    message?: string;
    model?: string;
    usage?: {
        input_tokens: number;
        output_tokens: number;
        total_tokens: number;
    };
    cost?: {
        input_cost: number;
        output_cost: number;
        total_cost: number;
    };
}

export class ExplainHtmlView extends Pane<PaneState> {
    private lastResult: CompilationResult | null = null;
    private loadingElement: JQuery;
    private consentElement: JQuery;
    private contentElement: JQuery;
    private bottomBarElement: JQuery;
    private statsElement: JQuery;
    private explainApiEndpoint: string;
    private fontScale: FontScale;

    // Use a static variable to persist consent across all instances during the session
    private static consentGiven = false;

    constructor(hub: Hub, container: Container, state: PaneState) {
        super(hub, container, state);
        this.explainApiEndpoint = (options.explainApiEndpoint as string) || 'https://api.compiler-explorer.com/explain';
        this.paneName = 'Claude Explain';
        this.updateTitle();

        this.loadingElement = this.domRoot.find('.explain-loading');
        this.consentElement = this.domRoot.find('.explain-consent');
        this.contentElement = this.domRoot.find('.explain-content');
        this.bottomBarElement = this.domRoot.find('.explain-bottom-bar');
        this.statsElement = this.domRoot.find('.explain-stats');

        // Setup font scale
        this.fontScale = new FontScale(this.domRoot, state, '.explain-content');
        this.fontScale.on('change', this.updateState.bind(this));

        // Register event handlers
        this.consentElement.find('.consent-btn').on('click', () => {
            ExplainHtmlView.consentGiven = true;
            this.consentElement.addClass('d-none');
            this.fetchExplanation();
        });
    }

    override getInitialHTML(): string {
        return $('#explain-view').html();
    }

    override onCompiler(
        compilerId: number,
        compiler: CompilerInfo | null,
        options: string,
        editorId: number,
        treeId: number,
    ): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        this.compilerInfo.compilerName = compiler ? compiler.name : '';
        this.compilerInfo.editorId = editorId;
        this.compilerInfo.treeId = treeId;
        this.updateTitle();
    }

    override onCompileResult(id: number, compiler: CompilerInfo, result: CompilationResult): void {
        try {
            if (id !== this.compilerInfo.compilerId) return;

            const foundTool = _.find(compiler.tools, tool => tool.tool.id === 'explain');
            const isToolAvailable = !!foundTool;
            this.toggleUsable(isToolAvailable);
            this.lastResult = result;

            if (!isToolAvailable) return;

            // Make sure title is correctly set
            this.paneName = 'Claude Explain';
            this.updateTitle();

            if (result.code !== 0) {
                // If compilation failed, show error message
                this.contentElement.text('Cannot explain: Compilation failed');
            } else if (ExplainHtmlView.consentGiven) {
                // Consent already given, fetch explanation automatically
                this.fetchExplanation();
            } else {
                // Show consent UI
                this.consentElement.removeClass('d-none');
                this.contentElement.text('Claude needs your consent to explain this code.');
            }
        } catch (e: any) {
            this.contentElement.text('javascript error: ' + e.message);
            SentryCapture(e);
        }
    }

    private toggleUsable(isUsable: boolean): void {
        if (isUsable) {
            this.contentElement.css('opacity', '1');
        } else {
            this.contentElement.css('opacity', '0.5');
        }
    }

    private showLoading(): void {
        this.loadingElement.removeClass('d-none');
        this.contentElement.text('Generating explanation...');
    }

    private hideLoading(): void {
        this.loadingElement.addClass('d-none');
    }

    private showBottomBar(): void {
        this.bottomBarElement.removeClass('d-none');
    }

    private updateStatsInBottomBar(data: ClaudeExplainResponse): void {
        if (!data.usage) return;

        const stats: string[] = [];
        if (data.model) {
            stats.push(`Model: ${data.model}`);
        }
        if (data.usage.total_tokens) {
            stats.push(`Tokens: ${data.usage.total_tokens}`);
        }
        if (data.cost?.total_cost) {
            stats.push(`Cost: $${data.cost.total_cost.toFixed(6)}`);
        }

        this.statsElement.text(stats.join(' | '));
    }

    private async fetchExplanation(): Promise<void> {
        if (!this.lastResult || !ExplainHtmlView.consentGiven) return;

        this.contentElement.empty();
        this.showLoading();

        try {
            // TODO: Improve language and instructionSet detection
            // Currently we're guessing language from compiler name and using a fixed architecture
            // We should get these from the proper compiler properties when available
            const payload = {
                source: this.lastResult.source || '',
                compiler: this.compilerInfo.compilerId.toString(),
                code: this.lastResult.source || '',
                compilationOptions: this.lastResult.compilationOptions || [],
                asm: this.lastResult.asm,
                instructionSet: 'amd64', // TODO: Get from compiler info when available
                language: this.compilerInfo.compilerName.split(' ')[0].toLowerCase(), // TODO: Get proper language
            };

            const response = await window.fetch(this.explainApiEndpoint, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status} ${response.statusText}`);
            }

            const data = (await response.json()) as ClaudeExplainResponse;
            this.hideLoading();

            if (data.status === 'error') {
                this.contentElement.text(`Error: ${data.message || 'Unknown error'}`);
                return;
            }

            // Render the markdown explanation
            this.renderMarkdown(data.explanation);

            // Show stats in bottom bar
            if (data.usage) {
                this.showBottomBar();
                this.updateStatsInBottomBar(data);
            }
        } catch (error) {
            this.hideLoading();
            this.contentElement.text(`Error: ${error instanceof Error ? error.message : String(error)}`);
            SentryCapture(error);
        }
    }

    private renderMarkdown(markdown: string): void {
        // Configure marked
        marked.setOptions({
            gfm: true, // GitHub Flavored Markdown
            breaks: true, // Convert line breaks to <br>
        });

        // Render markdown to HTML
        // marked.parse() is synchronous and returns a string, but TypeScript types suggest it could be Promise<string>
        // The cast is safe because we're using the default synchronous implementation
        this.contentElement.html(marked.parse(markdown) as string);
    }

    override resize(): void {
        // With flexbox layout, we don't need to manually calculate heights
        // Just ensure the topbar is properly sized
        utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);
    }

    override getDefaultPaneName(): string {
        return 'Claude Explain';
    }

    override close(): void {
        this.eventHub.unsubscribe();
    }
}
