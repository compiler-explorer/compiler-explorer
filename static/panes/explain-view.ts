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
import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {initPopover} from '../bootstrap-utils.js';
import {Hub} from '../hub.js';
import {options} from '../options.js';
import {SentryCapture} from '../sentry.js';
import * as utils from '../utils.js';
import {FontScale} from '../widgets/fontscale.js';
import {AvailableOptions, ClaudeExplainResponse, ExplainRequest, ExplainViewState} from './explain-view.interfaces.js';
import {Pane} from './pane.js';

export class ExplainView extends Pane<ExplainViewState> {
    private lastResult: CompilationResult | null = null;
    private compiler: CompilerInfo | null = null;
    private statusIcon: JQuery;
    private consentElement: JQuery;
    private noAiElement: JQuery;
    private contentElement: JQuery;
    private bottomBarElement: JQuery;
    private statsElement: JQuery;
    private audienceSelect: JQuery;
    private explanationSelect: JQuery;
    private audienceInfoButton: JQuery;
    private explanationInfoButton: JQuery;
    private explainApiEndpoint: string;
    private fontScale: FontScale;

    // Use a static variable to persist consent across all instances during the session
    private static consentGiven = false;

    // Static cache for available options (shared across all instances)
    private static availableOptions: AvailableOptions | null = null;
    private static optionsFetchPromise: Promise<AvailableOptions> | null = null;

    // Static cache for explanations (shared across all instances)
    private static cache: LRUCache<string, ClaudeExplainResponse> | null = null;

    // Instance variables for selected options
    private selectedAudience: string;
    private selectedExplanation: string;
    private isInitializing = true;

    constructor(hub: Hub, container: Container, state: ExplainViewState) {
        super(hub, container, state);
        // API endpoint from global options
        this.explainApiEndpoint = options.explainApiEndpoint || '';

        // Initialize static cache only once (shared across all instances)
        if (!ExplainView.cache) {
            ExplainView.cache = new LRUCache({
                maxSize: 200 * 1024,
                sizeCalculation: n => JSON.stringify(n).length,
            });
        }

        this.statusIcon = this.domRoot.find('.status-icon');
        this.consentElement = this.domRoot.find('.explain-consent');
        this.noAiElement = this.domRoot.find('.explain-no-ai');
        this.contentElement = this.domRoot.find('.explain-content');
        this.bottomBarElement = this.domRoot.find('.explain-bottom-bar');
        this.statsElement = this.domRoot.find('.explain-stats');
        this.audienceSelect = this.domRoot.find('.explain-audience');
        this.explanationSelect = this.domRoot.find('.explain-type');
        this.audienceInfoButton = this.domRoot.find('.explain-audience-info');
        this.explanationInfoButton = this.domRoot.find('.explain-type-info');

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

        // Wire up select controls
        this.audienceSelect.on('change', () => {
            this.selectedAudience = this.audienceSelect.val() as string;
            this.updateState();
            if (ExplainView.consentGiven && this.lastResult) {
                this.fetchExplanation();
            }
        });

        this.explanationSelect.on('change', () => {
            this.selectedExplanation = this.explanationSelect.val() as string;
            this.updateState();
            if (ExplainView.consentGiven && this.lastResult) {
                this.fetchExplanation();
            }
        });

        // Initialize UI controls
        this.initializeOptions();

        // Set initial content to avoid showing template content
        this.contentElement.text('Waiting for compilation...');
        this.isAwaitingInitialResults = true;

        // Emit explain view opened event
        this.eventHub.emit('explainViewOpened', this.compilerInfo.compilerId);
    }

    override getInitialHTML(): string {
        return $('#explain').html();
    }

    private async initializeOptions(): Promise<void> {
        try {
            const options = await this.fetchAvailableOptions();
            this.populateSelectOptions(options);
        } catch (error) {
            console.error('Failed to initialize options:', error);
            // Controls will remain with "Loading..." option
        }
    }

    private populateSelectOptions(options: AvailableOptions): void {
        // Populate audience select
        this.audienceSelect.empty();
        options.audience.forEach(option => {
            const optionElement = $('<option></option>')
                .attr('value', option.value)
                .text(option.value.charAt(0).toUpperCase() + option.value.slice(1))
                .attr('title', option.description);
            this.audienceSelect.append(optionElement);
        });

        // Populate explanation type select
        this.explanationSelect.empty();
        options.explanation.forEach(option => {
            const optionElement = $('<option></option>')
                .attr('value', option.value)
                .text(option.value.charAt(0).toUpperCase() + option.value.slice(1))
                .attr('title', option.description);
            this.explanationSelect.append(optionElement);
        });

        // Update popover content with the loaded options
        this.updatePopoverContent(options);

        if (this.isInitializing) {
            // During initialization: trust saved state completely, no validation
            this.audienceSelect.val(this.selectedAudience);
            this.explanationSelect.val(this.selectedExplanation);
            this.isInitializing = false; // Now user interactions can begin
        } else {
            // During runtime: validate user changes normally
            const validAudienceValue = options.audience.some(opt => opt.value === this.selectedAudience)
                ? this.selectedAudience
                : 'beginner';
            const validExplanationValue = options.explanation.some(opt => opt.value === this.selectedExplanation)
                ? this.selectedExplanation
                : 'assembly';

            this.selectedAudience = validAudienceValue;
            this.selectedExplanation = validExplanationValue;

            this.audienceSelect.val(validAudienceValue);
            this.explanationSelect.val(validExplanationValue);
        }
    }

    private updatePopoverContent(options: AvailableOptions): void {
        // Generate HTML content for audience popover
        const audienceContent = options.audience
            .map(
                option =>
                    `<div class="mb-2"><strong>${option.value.charAt(0).toUpperCase() + option.value.slice(1)}:</strong> ${option.description}</div>`,
            )
            .join('');

        // Generate HTML content for explanation popover
        const explanationContent = options.explanation
            .map(
                option =>
                    `<div class="mb-2"><strong>${option.value.charAt(0).toUpperCase() + option.value.slice(1)}:</strong> ${option.description}</div>`,
            )
            .join('');

        // Initialize Bootstrap popovers with the content
        initPopover(this.audienceInfoButton, {
            content: audienceContent,
            html: true,
            placement: 'bottom',
            trigger: 'focus',
        });

        initPopover(this.explanationInfoButton, {
            content: explanationContent,
            html: true,
            placement: 'bottom',
            trigger: 'focus',
        });
    }

    private async fetchAvailableOptions(): Promise<AvailableOptions> {
        // If we already have options cached, return them
        if (ExplainView.availableOptions) {
            return ExplainView.availableOptions;
        }

        // If we're already fetching, wait for that promise
        if (ExplainView.optionsFetchPromise) {
            return ExplainView.optionsFetchPromise;
        }

        // Create the fetch promise
        ExplainView.optionsFetchPromise = (async () => {
            try {
                const response = await window.fetch(this.explainApiEndpoint, {
                    method: 'GET',
                    headers: {'Content-Type': 'application/json'},
                });

                if (!response.ok) {
                    throw new Error(`Failed to fetch options: ${response.status} ${response.statusText}`);
                }

                const options = (await response.json()) as AvailableOptions;
                ExplainView.availableOptions = options;
                return options;
            } catch (error) {
                // If fetch fails, provide fallback options
                console.error('Failed to fetch available options:', error);
                const fallbackOptions: AvailableOptions = {
                    audience: [
                        {value: 'beginner', description: 'For beginners learning assembly language'},
                        {value: 'intermediate', description: 'For users familiar with basic assembly concepts'},
                        {value: 'expert', description: 'For advanced users'},
                    ],
                    explanation: [
                        {value: 'assembly', description: 'Explains the assembly instructions'},
                        {value: 'source', description: 'Explains how source code maps to assembly'},
                        {value: 'optimization', description: 'Explains compiler optimizations'},
                    ],
                };
                ExplainView.availableOptions = fallbackOptions;
                return fallbackOptions;
            } finally {
                // Clear the promise once done
                ExplainView.optionsFetchPromise = null;
            }
        })();

        return ExplainView.optionsFetchPromise;
    }

    override initializeStateDependentProperties(state: ExplainViewState): void {
        // Set defaults first
        this.selectedAudience = 'beginner';
        this.selectedExplanation = 'assembly';

        // Then override with saved state if it exists
        if (state.audience) {
            this.selectedAudience = state.audience;
        }
        if (state.explanation) {
            this.selectedExplanation = state.explanation;
        }
    }

    override getCurrentState(): ExplainViewState {
        const state = super.getCurrentState() as ExplainViewState;
        state.audience = this.selectedAudience;
        state.explanation = this.selectedExplanation;
        return state;
    }

    override onCompiler(
        compilerId: number,
        compiler: CompilerInfo | null,
        _compilerOptions: string,
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

            // Hide all special UI elements first
            this.consentElement.addClass('d-none');
            this.noAiElement.addClass('d-none');

            if (result.code !== 0) {
                // If compilation failed, show error message
                this.contentElement.text('Cannot explain: Compilation failed');
            } else if (result.source && this.checkForNoAiDirective(result.source)) {
                // Check for no-ai directive
                this.noAiElement.removeClass('d-none');
                this.contentElement.text('');
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

    private updateStatsInBottomBar(data: ClaudeExplainResponse, clientCacheHit = false, serverCacheHit = false): void {
        if (!data.usage) return;

        const stats: string[] = [];

        // Display cache status with appropriate icon
        if (clientCacheHit) {
            stats.push('🔄 Cached (client)');
        } else if (serverCacheHit) {
            stats.push('🔄 Cached (server)');
        } else {
            stats.push('✨ Fresh');
        }

        if (data.model) {
            stats.push(`Model: ${data.model}`);
        }
        if (data.usage.totalTokens) {
            stats.push(`Tokens: ${data.usage.totalTokens}`);
        }
        if (data.cost?.totalCost) {
            stats.push(`Cost: $${data.cost.totalCost.toFixed(6)}`);
        }

        this.statsElement.text(stats.join(' | '));
    }

    private generateCacheKey(payload: ExplainRequest): string {
        // Create a cache key from the request payload
        // Sort the payload properties to ensure consistent key generation
        const sortedPayload = {
            language: payload.language,
            compiler: payload.compiler,
            code: payload.code,
            compilationOptions: payload.compilationOptions?.sort() || [],
            instructionSet: payload.instructionSet,
            asm: payload.asm,
            audience: payload.audience || 'beginner',
            explanation: payload.explanation || 'assembly',
        };
        return JSON.stringify(sortedPayload);
    }

    private checkForNoAiDirective(sourceCode: string): boolean {
        // Check for no-ai directive (case insensitive)
        return /no-ai/i.test(sourceCode);
    }

    private async fetchExplanation(bypassCache = false): Promise<void> {
        if (!this.lastResult || !ExplainView.consentGiven || !this.compiler) return;

        if (!this.explainApiEndpoint) {
            this.contentElement.text('Error: Claude Explain API endpoint not configured');
            return;
        }

        // Check for no-ai directive in source code
        if (this.lastResult.source && this.checkForNoAiDirective(this.lastResult.source)) {
            this.hideLoading();
            this.noAiElement.removeClass('d-none');
            this.contentElement.text('');
            return;
        }

        this.contentElement.empty();
        this.showLoading();

        try {
            const payload: ExplainRequest = {
                language: this.compiler.lang,
                compiler: this.compiler.name,
                code: this.lastResult.source || '',
                compilationOptions: this.lastResult.compilationOptions || [],
                instructionSet: this.lastResult.instructionSet || 'amd64',
                asm: Array.isArray(this.lastResult.asm) ? this.lastResult.asm : [],
                audience: this.selectedAudience,
                explanation: this.selectedExplanation,
            };

            // Add bypassCache flag if requested
            if (bypassCache) {
                payload.bypassCache = true;
            }

            const cacheKey = this.generateCacheKey(payload);

            // Check cache first unless bypassing
            if (!bypassCache) {
                const cachedResult = ExplainView.cache?.get(cacheKey);
                if (cachedResult) {
                    this.hideLoading();
                    this.showSuccess();
                    // Render the cached explanation
                    this.renderMarkdown(cachedResult.explanation);

                    // Show bottom bar with reload button
                    this.showBottomBar();
                    // Show stats if available
                    if (cachedResult.usage) {
                        this.updateStatsInBottomBar(cachedResult, true, false);
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
            ExplainView.cache?.set(cacheKey, data);

            // Render the markdown explanation
            this.renderMarkdown(data.explanation);

            // Show bottom bar with reload button
            this.showBottomBar();
            // Show stats if available
            if (data.usage) {
                // Pass server cache status from response
                this.updateStatsInBottomBar(data, false, data.cached);
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
