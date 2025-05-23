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
import {LRUCache} from 'lru-cache';
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

// TODO: Improvement opportunities for ExplainView:
// 1. Extract loading state management (showLoading/hideLoading) to base class or mixin
// 2. Create explain-view.interfaces.ts file for consistency with other panes
// 3. Extract markdown styles to shared markdown.scss (221 lines of duplication)
// 5. Consider state machine pattern for clearer UI state transitions
// 6. Add tests: frontend tests for ExplainView
// 7. Improve error handling with different UI states for different error types
// 9. Apply theming correctly (pink mode is broken)
// 10. Address TODOs in the documentation, and tidy that up too.
export class ExplainView extends Pane<PaneState> {
    private lastResult: CompilationResult | null = null;
    private compiler: CompilerInfo | null = null;
    private statusIcon: JQuery;
    private consentElement: JQuery;
    private contentElement: JQuery;
    private bottomBarElement: JQuery;
    private statsElement: JQuery;
    private explainApiEndpoint: string;
    private fontScale: FontScale;
    private cache: LRUCache<string, ClaudeExplainResponse>;

    // Use a static variable to persist consent across all instances during the session
    private static consentGiven = false;

    constructor(hub: Hub, container: Container, state: PaneState) {
        super(hub, container, state);
        // API endpoint from global options
        this.explainApiEndpoint = options.explainApiEndpoint || '';

        // Initialize cache with same settings as CompilerService
        this.cache = new LRUCache({
            maxSize: 200 * 1024,
            sizeCalculation: n => JSON.stringify(n).length,
        });

        this.statusIcon = this.domRoot.find('.status-icon');
        this.consentElement = this.domRoot.find('.explain-consent');
        this.contentElement = this.domRoot.find('.explain-content');
        this.bottomBarElement = this.domRoot.find('.explain-bottom-bar');
        this.statsElement = this.domRoot.find('.explain-stats');

        this.fontScale = new FontScale(this.domRoot, state, '.explain-content');
        this.fontScale.on('change', this.updateState.bind(this));

        this.consentElement.find('.consent-btn').on('click', () => {
            ExplainView.consentGiven = true;
            this.consentElement.addClass('d-none');
            this.fetchExplanation();
        });

        // Wire up reload button to bypass cache
        this.bottomBarElement.find('.explain-reload').on('click', () => {
            this.fetchExplanation(true);
        });

        // Set initial content to avoid showing template content
        this.contentElement.text('Waiting for compilation...');
        this.isAwaitingInitialResults = true;

        // Emit explain view opened event
        this.eventHub.emit('explainViewOpened', this.compilerInfo.compilerId);
    }

    override getInitialHTML(): string {
        return $('#explain').html();
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
        this.compiler = compiler;
        this.updateTitle();
    }

    override onCompileResult(id: number, compiler: CompilerInfo, result: CompilationResult): void {
        try {
            if (id !== this.compilerInfo.compilerId) return;
            this.compiler = compiler;

            this.lastResult = result;

            // Mark that we've received our first result
            this.isAwaitingInitialResults = false;

            if (result.code !== 0) {
                // If compilation failed, show error message
                this.contentElement.text('Cannot explain: Compilation failed');
            } else if (ExplainView.consentGiven) {
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

    private showLoading(): void {
        this.statusIcon
            .removeClass()
            .addClass('status-icon fas fa-spinner fa-spin')
            .css('color', '')
            .attr('aria-label', 'Generating explanation...');
        this.contentElement.text('Generating explanation...');
    }

    private hideLoading(): void {
        this.statusIcon.removeClass().addClass('status-icon fas d-none');
    }

    private showSuccess(): void {
        this.statusIcon
            .removeClass()
            .addClass('status-icon fas fa-check-circle')
            .css('color', '#4CAF50')
            .attr('aria-label', 'Explanation generated successfully');
    }

    private showError(): void {
        this.statusIcon
            .removeClass()
            .addClass('status-icon fas fa-times-circle')
            .css('color', '#FF6645')
            .attr('aria-label', 'Error generating explanation');
    }

    private showBottomBar(): void {
        this.bottomBarElement.removeClass('d-none');
    }

    private updateStatsInBottomBar(data: ClaudeExplainResponse, cacheHit = false): void {
        if (!data.usage) return;

        const stats: string[] = [];
        if (cacheHit) {
            stats.push('(Cached)');
        }
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

    private generateCacheKey(payload: any): string {
        // Create a cache key from the request payload
        // Sort the payload properties to ensure consistent key generation
        const sortedPayload = {
            language: payload.language,
            compiler: payload.compiler,
            code: payload.code,
            compilationOptions: payload.compilationOptions?.sort() || [],
            instructionSet: payload.instructionSet,
            asm: payload.asm,
        };
        return JSON.stringify(sortedPayload);
    }

    private async fetchExplanation(bypassCache = false): Promise<void> {
        if (!this.lastResult || !ExplainView.consentGiven || !this.compiler) return;

        if (!this.explainApiEndpoint) {
            this.contentElement.text('Error: Claude Explain API endpoint not configured');
            return;
        }

        this.contentElement.empty();
        this.showLoading();

        try {
            const payload = {
                language: this.compiler.lang,
                compiler: this.compiler.name,
                code: this.lastResult.source || '',
                compilationOptions: this.lastResult.compilationOptions || [],
                instructionSet: this.lastResult.instructionSet || 'amd64',
                asm: this.lastResult.asm,
            };

            const cacheKey = this.generateCacheKey(payload);

            // Check cache first unless bypassing
            if (!bypassCache) {
                const cachedResult = this.cache.get(cacheKey);
                if (cachedResult) {
                    this.hideLoading();
                    this.showSuccess();
                    // Render the cached explanation
                    this.renderMarkdown(cachedResult.explanation);

                    // Show bottom bar with reload button
                    this.showBottomBar();
                    // Show stats if available
                    if (cachedResult.usage) {
                        this.updateStatsInBottomBar(cachedResult, true);
                    }
                    return;
                }
            }

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
                this.showError();
                this.contentElement.text(`Error: ${data.message || 'Unknown error'}`);
                return;
            }

            this.showSuccess();

            // Cache the successful response
            this.cache.set(cacheKey, data);

            // Render the markdown explanation
            this.renderMarkdown(data.explanation);

            // Show bottom bar with reload button
            this.showBottomBar();
            // Show stats if available
            if (data.usage) {
                this.updateStatsInBottomBar(data);
            }
        } catch (error) {
            this.hideLoading();
            this.showError();
            this.contentElement.text(`Error: ${error instanceof Error ? error.message : String(error)}`);
            SentryCapture(error);
        }
    }

    private renderMarkdown(markdown: string): void {
        const markedOptions = {
            gfm: true, // GitHub Flavored Markdown
            breaks: true, // Convert line breaks to <br>
        };

        // Render markdown to HTML
        // marked.parse() is synchronous and returns a string, but TypeScript types suggest it could be Promise<string>
        // The cast is safe because we're using the default synchronous implementation
        this.contentElement.html(marked.parse(markdown, markedOptions) as string);
    }

    override resize(): void {
        utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);
    }

    override getDefaultPaneName(): string {
        return 'Claude Explain';
    }

    override close(): void {
        this.eventHub.emit('explainViewClosed', this.compilerInfo.compilerId);
        this.eventHub.unsubscribe();
    }
}
