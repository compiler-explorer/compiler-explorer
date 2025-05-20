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
import _ from 'underscore';
import {CompilationResult} from '../compilation/compilation.interfaces.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {ToolState} from '../components.interfaces.js';
import {Hub} from '../hub.js';
import {options} from '../options.js';
import {SentryCapture} from '../sentry.js';
import {MonacoPaneState} from './pane.interfaces.js';
import {Tool} from './tool.js';

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

export class ExplainView extends Tool {
    private lastResult: CompilationResult | null = null;
    private loadingElement: JQuery;
    private consentElement: JQuery;
    private explainApiEndpoint: string;

    // Use a static variable to persist consent across all instances during the session
    private static consentGiven = false;

    constructor(hub: Hub, container: Container, state: ToolState & MonacoPaneState) {
        super(hub, container, state);
        this.explainApiEndpoint = (options.explainApiEndpoint as string) || 'https://api.compiler-explorer.com/explain';
        this.toolName = 'Claude Explain';
        this.updateTitle(); // Call updateTitle after setting toolName

        this.loadingElement = this.domRoot.find('.explain-loading');
        if (this.loadingElement.length === 0) {
            this.loadingElement = $('<div class="explain-loading d-none">Generating explanation...</div>');
            this.domRoot.find('.content').before(this.loadingElement);
        }

        this.consentElement = this.domRoot.find('.explain-consent');
        if (this.consentElement.length === 0) {
            this.consentElement = $(
                '<div class="explain-consent">' +
                    '<p>Claude Explain will send your code and compilation output to an external API. ' +
                    'Continue?</p>' +
                    '<button class="btn btn-primary consent-btn">Yes, explain this code</button>' +
                    '</div>',
            );
            this.domRoot.find('.content').before(this.consentElement);

            this.consentElement.find('.consent-btn').on('click', () => {
                ExplainView.consentGiven = true;
                this.consentElement.addClass('d-none');
                this.fetchExplanation();
            });
        }
    }

    override onCompileResult(id: number, compiler: CompilerInfo, result: CompilationResult) {
        try {
            if (id !== this.compilerInfo.compilerId) return;

            const foundTool = _.find(compiler.tools, tool => tool.tool.id === this.toolId);
            this.toggleUsable(!!foundTool);
            this.lastResult = result;

            // If this is the explain tool, show the consent UI or fetch explanation
            if (this.toolId === 'explain') {
                // Make sure title is correctly set
                this.toolName = 'Claude Explain';
                this.updateTitle();

                if (result.code !== 0) {
                    // If compilation failed, show error message
                    this.setLanguage(false);
                    this.add('Cannot explain: Compilation failed');
                } else if (ExplainView.consentGiven) {
                    // Consent already given, fetch explanation automatically
                    this.fetchExplanation();
                } else {
                    // Show consent UI
                    this.consentElement.removeClass('d-none');
                    this.setLanguage(false);
                    this.add('Claude needs your consent to explain this code.');
                }
            } else {
                // For non-explain tools, use the default behavior
                super.onCompileResult(id, compiler, result);
            }
        } catch (e: any) {
            this.setLanguage(false);
            this.add('javascript error: ' + e.message);
        }
    }

    private showLoading() {
        this.loadingElement.removeClass('d-none');
        this.setLanguage(false);
        this.add('Generating explanation...');
    }

    private hideLoading() {
        this.loadingElement.addClass('d-none');
    }

    private async fetchExplanation() {
        if (!this.lastResult || !ExplainView.consentGiven) return;

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

            const data = (await response.json()) as ClaudeExplainResponse;
            this.hideLoading();

            if (data.status === 'error') {
                this.setLanguage(false);
                this.add(`Error: ${data.message || 'Unknown error'}`);
                return;
            }

            // Set language to markdown for proper rendering
            this.setLanguage('markdown');
            this.setEditorContent(data.explanation);

            // Add usage info if available
            if (data.usage) {
                const usageInfo = `\n\n---\n\n*Tokens: ${data.usage.total_tokens} | Model: ${data.model || 'Claude'}*`;
                this.setEditorContent(this.editor.getValue() + usageInfo);
            }
        } catch (error) {
            this.hideLoading();
            this.setLanguage(false);
            this.add(`Failed to get explanation: ${error instanceof Error ? error.message : String(error)}`);
            SentryCapture(error);
        }
    }
}
