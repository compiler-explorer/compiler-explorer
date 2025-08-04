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
import {capitaliseFirst} from '../../shared/common-utils.js';
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

enum StatusIconState {
    Loading = 'loading',
    Success = 'success',
    Error = 'error',
    Hidden = 'hidden',
}

const defaultAudienceType = 'beginner';
const defaultExplanationType = 'assembly';

const statusIconConfigs = {
    [StatusIconState.Loading]: {
        classes: 'status-icon fas fa-spinner fa-spin',
        color: '',
        ariaLabel: 'Generating explanation...',
    },
    [StatusIconState.Success]: {
        classes: 'status-icon fas fa-check-circle',
        color: '#4CAF50',
        ariaLabel: 'Explanation generated successfully',
    },
    [StatusIconState.Error]: {
        classes: 'status-icon fas fa-times-circle',
        color: '#FF6645',
        ariaLabel: 'Error generating explanation',
    },
    [StatusIconState.Hidden]: {
        classes: 'status-icon fas d-none',
        color: '',
        ariaLabel: '',
    },
} as const;

export class ExplainView extends Pane<ExplainViewState> {
    private lastResult: CompilationResult | null = null;
    private compiler: CompilerInfo | null = null;
    private readonly statusIcon: JQuery;
    private readonly consentElement: JQuery;
    private readonly noAiElement: JQuery;
    private readonly contentElement: JQuery;
    private readonly bottomBarElement: JQuery;
    private readonly statsElement: JQuery;
    private readonly audienceSelect: JQuery;
    private readonly explanationSelect: JQuery;
    private readonly audienceInfoButton: JQuery;
    private readonly explanationInfoButton: JQuery;
    private readonly explainApiEndpoint: string;
    private readonly fontScale: FontScale;
    private readonly cache: LRUCache<string, ClaudeExplainResponse>;

    // Use a static variable to persist consent across all instances during the session
    private static consentGiven = false;

    // Static cache for available options (shared across all instances)
    private static availableOptions: AvailableOptions | null = null;
    private static optionsFetchPromise: Promise<AvailableOptions> | null = null;

    // Instance variables for selected options
    private selectedAudience: string;
    private selectedExplanation: string;
    private isInitializing = true;

    constructor(hub: Hub, container: Container, state: ExplainViewState) {
        super(hub, container, state);

        // Initialize properties
        this.explainApiEndpoint = options.explainApiEndpoint ?? '';
        this.cache = new LRUCache({
            maxSize: 200 * 1024,
            sizeCalculation: n => JSON.stringify(n).length,
        });

        // Setup UI elements
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

        this.attachEventListeners();
        void this.initializeOptions();

        this.contentElement.text('Waiting for compilation...');
        this.isAwaitingInitialResults = true;
        this.eventHub.emit('explainViewOpened', this.compilerInfo.compilerId);
    }

    private handleConsentClick(): void {
        ExplainView.consentGiven = true;
        this.consentElement.addClass('d-none');
        void this.fetchExplanation();
    }

    private handleReloadClick(): void {
        void this.fetchExplanation(true);
    }

    private handleAudienceChange(): void {
        this.selectedAudience = this.audienceSelect.val() as string;
        this.updateState();
        this.refreshExplanationIfReady();
    }

    private handleExplanationChange(): void {
        this.selectedExplanation = this.explanationSelect.val() as string;
        this.updateState();
        this.refreshExplanationIfReady();
    }

    private refreshExplanationIfReady(): void {
        if (ExplainView.consentGiven && this.lastResult) {
            void this.fetchExplanation();
        }
    }

    private attachEventListeners(): void {
        this.consentElement.find('.consent-btn').on('click', this.handleConsentClick.bind(this));
        this.bottomBarElement.find('.explain-reload').on('click', this.handleReloadClick.bind(this));
        this.audienceSelect.on('change', this.handleAudienceChange.bind(this));
        this.explanationSelect.on('change', this.handleExplanationChange.bind(this));
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
            this.showExplainUnavailable();
        }
    }

    private showExplainUnavailable(): void {
        const emptyOptions = [{value: '', description: 'Service unavailable'}];
        this.populateSelect(this.audienceSelect, emptyOptions);
        this.populateSelect(this.explanationSelect, emptyOptions);

        this.audienceSelect.prop('disabled', true);
        this.explanationSelect.prop('disabled', true);

        this.contentElement.html(
            '<div class="alert alert-warning">Claude Explain is currently unavailable due to a service error. Please try again later.</div>',
        );
    }

    private populateSelect(selectElement: JQuery, optionsList: Array<{value: string; description: string}>): void {
        selectElement.empty();
        optionsList.forEach(option => {
            const optionElement = $('<option></option>')
                .attr('value', option.value)
                .text(capitaliseFirst(option.value))
                .attr('title', option.description);
            selectElement.append(optionElement);
        });
    }

    private populateSelectOptions(options: AvailableOptions): void {
        this.populateSelect(this.audienceSelect, options.audience);
        this.populateSelect(this.explanationSelect, options.explanation);
        this.updatePopoverContent(options);

        if (this.isInitializing) {
            // During initialisation: trust saved state completely, no validation
            this.audienceSelect.val(this.selectedAudience);
            this.explanationSelect.val(this.selectedExplanation);
            this.isInitializing = false; // Now user interactions can begin
        } else {
            // After initialisation: validate user changes normally
            const validAudienceValue = options.audience.some(opt => opt.value === this.selectedAudience)
                ? this.selectedAudience
                : defaultAudienceType;
            const validExplanationValue = options.explanation.some(opt => opt.value === this.selectedExplanation)
                ? this.selectedExplanation
                : defaultExplanationType;

            this.selectedAudience = validAudienceValue;
            this.selectedExplanation = validExplanationValue;

            this.audienceSelect.val(validAudienceValue);
            this.explanationSelect.val(validExplanationValue);
        }
    }

    private createPopoverContent(optionsList: Array<{value: string; description: string}>): string {
        return optionsList
            .map(
                option =>
                    `<div class='mb-2'><strong>${capitaliseFirst(option.value)}:</strong> ${option.description}</div>`,
            )
            .join('');
    }

    private updatePopoverContent(options: AvailableOptions): void {
        const audienceContent = this.createPopoverContent(options.audience);
        const explanationContent = this.createPopoverContent(options.explanation);

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
        if (ExplainView.availableOptions) return ExplainView.availableOptions;

        // If we're already fetching, wait for that promise
        if (ExplainView.optionsFetchPromise) return ExplainView.optionsFetchPromise;

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
                // If fetch fails, propagate the error
                console.error('Failed to fetch available options:', error);
                throw error;
            } finally {
                // Clear the promise once done
                ExplainView.optionsFetchPromise = null;
            }
        })();

        return ExplainView.optionsFetchPromise;
    }

    override initializeStateDependentProperties(state: ExplainViewState): void {
        this.selectedAudience = state.audience ?? 'beginner';
        this.selectedExplanation = state.explanation ?? 'assembly';
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

    private hideSpecialUIElements(): void {
        this.consentElement.addClass('d-none');
        this.noAiElement.addClass('d-none');
    }

    private handleCompilationResult(result: CompilationResult): void {
        // If options failed to load, explain is unavailable
        if (ExplainView.availableOptions === null) {
            // Don't override the error message already shown
            return;
        }

        if (result.code !== 0) {
            this.contentElement.text('Cannot explain: Compilation failed');
            return;
        }

        if (result.source && this.checkForNoAiDirective(result.source)) {
            this.showNoAiDirective();
            return;
        }

        if (ExplainView.consentGiven) {
            void this.fetchExplanation();
        } else {
            this.showConsentUI();
        }
    }

    private showNoAiDirective(): void {
        this.noAiElement.removeClass('d-none');
        this.contentElement.text('');
    }

    private showConsentUI(): void {
        this.consentElement.removeClass('d-none');
        this.contentElement.text('Claude needs your consent to explain this code.');
    }

    override onCompileResult(id: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (id !== this.compilerInfo.compilerId) return;

        this.compiler = compiler;
        this.lastResult = result;
        this.isAwaitingInitialResults = false;

        this.hideSpecialUIElements();
        this.handleCompilationResult(result);
    }

    private setStatusIcon(state: StatusIconState): void {
        const config = statusIconConfigs[state];
        this.statusIcon
            .removeClass()
            .addClass(config.classes)
            .css('color', config.color)
            .attr('aria-label', config.ariaLabel);
    }

    private showLoading(): void {
        this.setStatusIcon(StatusIconState.Loading);
        this.contentElement.text('Generating explanation...');
    }

    private hideLoading(): void {
        this.setStatusIcon(StatusIconState.Hidden);
    }

    private showSuccess(): void {
        this.setStatusIcon(StatusIconState.Success);
    }

    private showError(): void {
        this.setStatusIcon(StatusIconState.Error);
    }

    private showBottomBar(): void {
        this.bottomBarElement.removeClass('d-none');
    }

    private updateStatsInBottomBar(data: ClaudeExplainResponse, clientCacheHit = false, serverCacheHit = false): void {
        if (!data.usage) return;

        const stats: string[] = [];

        // Display cache status with appropriate icon
        if (clientCacheHit) {
            stats.push('Cached (client)');
        } else if (serverCacheHit) {
            stats.push('Cached (server)');
        } else {
            stats.push('Fresh');
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
        // Create a cache key from the request payload.
        return JSON.stringify({
            language: payload.language,
            compiler: payload.compiler,
            code: payload.code,
            compilationOptions: payload.compilationOptions ?? [],
            instructionSet: payload.instructionSet,
            asm: payload.asm,
            audience: payload.audience,
            explanation: payload.explanation,
        });
    }

    private checkForNoAiDirective(sourceCode: string): boolean {
        return /no-ai/i.test(sourceCode);
    }

    private displayCachedResult(cachedResult: ClaudeExplainResponse): void {
        this.hideLoading();
        this.showSuccess();
        this.renderMarkdown(cachedResult.explanation);
        this.showBottomBar();

        if (cachedResult.usage) {
            this.updateStatsInBottomBar(cachedResult, true, false);
        }
    }

    private async fetchFromAPI(payload: ExplainRequest): Promise<ClaudeExplainResponse> {
        const response = await window.fetch(this.explainApiEndpoint, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status} ${response.statusText}`);
        }

        return response.json() as Promise<ClaudeExplainResponse>;
    }

    private validateFetchPreconditions(): void {
        if (!this.lastResult || !ExplainView.consentGiven || !this.compiler) {
            throw new Error('Missing required data: compilation result, consent, or compiler info');
        }

        if (ExplainView.availableOptions === null) {
            throw new Error('Explain options not available');
        }

        if (!this.explainApiEndpoint) {
            throw new Error('Claude Explain API endpoint not configured');
        }

        if (this.lastResult.source && this.checkForNoAiDirective(this.lastResult.source)) {
            throw new Error('no-ai directive found in source code');
        }
    }

    private buildExplainRequest(bypassCache: boolean): ExplainRequest {
        if (!this.compiler || !this.lastResult) {
            throw new Error('Missing compiler or compilation result');
        }

        return {
            language: this.compiler.lang,
            compiler: this.compiler.name,
            code: this.lastResult.source ?? '',
            compilationOptions: this.lastResult.compilationOptions ?? [],
            instructionSet: this.lastResult.instructionSet ?? 'amd64',
            asm: Array.isArray(this.lastResult.asm) ? this.lastResult.asm : [],
            audience: this.selectedAudience,
            explanation: this.selectedExplanation,
            ...(bypassCache && {bypassCache: true}),
        };
    }

    private checkAndReturnCachedResult(cacheKey: string): ClaudeExplainResponse | null {
        return this.cache.get(cacheKey) ?? null;
    }

    private processExplanationResponse(data: ClaudeExplainResponse, cacheKey: string): void {
        this.hideLoading();

        if (data.status === 'error') {
            this.showError();
            this.contentElement.text(`Error: ${data.message || 'Unknown error'}`);
            return;
        }

        this.showSuccess();
        this.cache.set(cacheKey, data);
        this.renderMarkdown(data.explanation);
        this.showBottomBar();

        if (data.usage) {
            this.updateStatsInBottomBar(data, false, data.cached);
        }
    }

    private async fetchExplanation(bypassCache = false): Promise<void> {
        try {
            this.validateFetchPreconditions();
        } catch (error) {
            if (error instanceof Error && error.message === 'no-ai directive found in source code') {
                this.hideLoading();
                this.noAiElement.removeClass('d-none');
                this.contentElement.text('');
                return;
            }
            if (error instanceof Error && error.message === 'Claude Explain API endpoint not configured') {
                this.contentElement.text('Error: Claude Explain API endpoint not configured');
                return;
            }
            // For other validation errors, just return silently
            return;
        }

        this.contentElement.empty();
        this.showLoading();

        try {
            const payload = this.buildExplainRequest(bypassCache);
            const cacheKey = this.generateCacheKey(payload);

            // Check cache first unless bypassing
            if (!bypassCache) {
                const cachedResult = this.checkAndReturnCachedResult(cacheKey);
                if (cachedResult) {
                    this.displayCachedResult(cachedResult);
                    return;
                }
            }

            const data = await this.fetchFromAPI(payload);
            this.processExplanationResponse(data, cacheKey);
        } catch (error) {
            this.handleFetchError(error);
        }
    }

    private handleFetchError(error: unknown): void {
        this.hideLoading();
        this.showError();
        const errorMessage = error instanceof Error ? error.message : String(error);
        this.contentElement.text(`Error: ${errorMessage}`);
        SentryCapture(error);
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
