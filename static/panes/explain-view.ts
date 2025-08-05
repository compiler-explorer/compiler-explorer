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
import {
    buildExplainRequest,
    checkForNoAiDirective,
    createPopoverContent,
    ExplainContext,
    formatErrorMessage,
    formatMarkdown,
    formatStatsText,
    generateCacheKey,
    ValidationErrorCode,
    validateExplainPreconditions,
} from './explain-view-utils.js';
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
    // Use a static variable to persist consent across all instances during the session
    private static consentGiven = false;

    // Static cache for available options (shared across all instances)
    private static availableOptions: AvailableOptions | null = null;
    private static optionsFetchPromise: Promise<AvailableOptions> | null = null;

    // Static explanation cache shared across all instances (200KB limit)
    private static cache: LRUCache<string, ClaudeExplainResponse> | null = null;

    // Instance variables for selected options
    private selectedAudience: string;
    private selectedExplanation: string;
    private isInitializing = true;

    // Store compilation results that arrive before initialization completes
    private pendingCompilationResult: CompilationResult | null = null;

    constructor(hub: Hub, container: Container, state: ExplainViewState) {
        super(hub, container, state);

        this.explainApiEndpoint = options.explainApiEndpoint ?? '';

        // Initialize static cache if not already done
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
            this.populateSelectOptions(await this.fetchAvailableOptions());
            this.isInitializing = false;

            // Process any compilation results that arrived while we were initializing.
            if (this.pendingCompilationResult) {
                this.handleCompilationResult(this.pendingCompilationResult);
                this.pendingCompilationResult = null;
            }
        } catch (error) {
            this.isInitializing = false;
            console.error('Failed to initialize options:', error);
            this.showExplainUnavailable();
            // Even if initialization failed, clear any pending results
            this.pendingCompilationResult = null;
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

    private updatePopoverContent(options: AvailableOptions): void {
        const audienceContent = createPopoverContent(options.audience);
        const explanationContent = createPopoverContent(options.explanation);

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

        // Else, go fetch the options
        ExplainView.optionsFetchPromise = (async () => {
            try {
                const response = await fetch(this.explainApiEndpoint, {
                    method: 'GET',
                    headers: {'Content-Type': 'application/json'},
                });

                if (!response.ok) {
                    throw new Error(`Failed to fetch options: ${response.status} ${response.statusText}`);
                }

                const options = (await response.json()) as AvailableOptions;
                ExplainView.availableOptions = options;
                return options;
            } finally {
                ExplainView.optionsFetchPromise = null;
            }
        })();

        return ExplainView.optionsFetchPromise;
    }

    override initializeStateDependentProperties(state: ExplainViewState): void {
        this.selectedAudience = state.audience ?? defaultAudienceType;
        this.selectedExplanation = state.explanation ?? defaultExplanationType;
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
        if (this.isInitializing) {
            // Store for processing after initialization completes
            this.pendingCompilationResult = result;
            return;
        }

        if (result.code !== 0) {
            this.contentElement.text('Cannot explain: Compilation failed');
            return;
        }

        if (result.source && checkForNoAiDirective(result.source)) {
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

    private updateStatsInBottomBar(
        data: ClaudeExplainResponse,
        clientCacheHit: boolean,
        serverCacheHit: boolean,
    ): void {
        const stats = formatStatsText(data, clientCacheHit, serverCacheHit);
        if (stats.length > 0) {
            this.statsElement.text(stats.join(' | '));
        }
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

    private getExplainContext(): ExplainContext {
        return {
            lastResult: this.lastResult,
            compiler: this.compiler,
            selectedAudience: this.selectedAudience,
            selectedExplanation: this.selectedExplanation,
            explainApiEndpoint: this.explainApiEndpoint,
            consentGiven: ExplainView.consentGiven,
            availableOptions: ExplainView.availableOptions,
        };
    }

    private processExplanationResponse(data: ClaudeExplainResponse, cacheKey: string): void {
        this.hideLoading();

        if (data.status === 'error') {
            this.showError();
            this.contentElement.text(`Error: ${data.message || 'Unknown error'}`);
            return;
        }

        this.showSuccess();
        ExplainView.cache!.set(cacheKey, data);
        this.renderMarkdown(data.explanation);
        this.showBottomBar();

        if (data.usage) {
            this.updateStatsInBottomBar(data, false, data.cached);
        }
    }

    private async fetchExplanation(bypassCache = false): Promise<void> {
        const context = this.getExplainContext();
        const validationResult = validateExplainPreconditions(context);

        if (!validationResult.success) {
            switch (validationResult.errorCode) {
                case ValidationErrorCode.NO_AI_DIRECTIVE_FOUND:
                    this.hideLoading();
                    this.noAiElement.removeClass('d-none');
                    this.contentElement.text('');
                    break;
                case ValidationErrorCode.MISSING_REQUIRED_DATA:
                    // Silent return - this is expected during normal UI flow (before compilation, before consent, etc.)
                    break;
                default:
                    // Show all other validation errors to help with debugging
                    this.contentElement.text(`Error: ${validationResult.message}`);
                    break;
            }
            return;
        }

        this.contentElement.empty();
        this.showLoading();

        try {
            const payload = buildExplainRequest(context, bypassCache);
            const cacheKey = generateCacheKey(payload);

            // Check cache first unless bypassing
            if (!bypassCache) {
                const cachedResult = ExplainView.cache!.get(cacheKey);
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
        this.contentElement.text(formatErrorMessage(error));
        SentryCapture(error);
    }

    private renderMarkdown(markdown: string): void {
        this.contentElement.html(formatMarkdown(markdown));
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
