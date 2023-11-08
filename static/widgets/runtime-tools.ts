// Copyright (c) 2023, Compiler Explorer Authors
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
import {options} from '../options.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {assert} from '../assert.js';
import {localStorage} from '../local.js';
import {
    ConfiguredRuntimeTool,
    ConfiguredRuntimeTools,
    PossibleRuntimeTools,
    RuntimeToolOption,
    RuntimeToolOptions,
    RuntimeToolType,
} from '../../types/execution/execution.interfaces.js';

const FAV_RUNTIMETOOLS_STORE_KEY = 'favruntimetools';

export type RuntimeToolsChangeCallback = () => void;

type FavRuntimeTool = {
    name: RuntimeToolType;
    options: string;
    meta: string;
};

type FavRuntimeTools = FavRuntimeTool[];

export class RuntimeToolsWidget {
    private domRoot: JQuery;
    private popupDomRoot: JQuery<HTMLElement>;
    private envVarsInput: JQuery<HTMLElement>;
    private dropdownButton: JQuery;
    private onChangeCallback: RuntimeToolsChangeCallback;
    private configured: ConfiguredRuntimeTools = [];
    private compiler: CompilerInfo | undefined;
    private possibleTools: PossibleRuntimeTools;

    constructor(domRoot: JQuery, dropdownButton: JQuery, onChangeCallback: RuntimeToolsChangeCallback) {
        this.domRoot = domRoot;
        this.popupDomRoot = $('#runtimetools-selection');
        this.dropdownButton = dropdownButton;
        this.envVarsInput = this.popupDomRoot.find('.envvars');
        this.onChangeCallback = onChangeCallback;
        this.possibleTools = [];
    }

    private loadStateFromUI(): ConfiguredRuntimeTools {
        const tools: ConfiguredRuntimeTools = [];

        const envOverrides = this.getEnvOverrides();
        if (envOverrides.length > 0) {
            tools.push({
                name: RuntimeToolType.env,
                options: envOverrides,
            });
        }

        const selects = this.popupDomRoot.find('select');
        for (const select of selects) {
            const jqSelect = $(select);

            const rawName = jqSelect.data('tool-name');
            const optionName = jqSelect.data('tool-option');

            const val = jqSelect.val();
            if (val) {
                const name = rawName as RuntimeToolType;
                assert(name !== RuntimeToolType.env);
                let tool = tools.find(tool => tool.name === name);
                if (!tool) {
                    tool = {
                        name: name,
                        options: [],
                    };
                    tools.push(tool);
                }

                const option: RuntimeToolOption = {
                    name: optionName,
                    value: (val || '') as string,
                };
                tool.options.push(option);
            }
        }

        return tools;
    }

    private optionsToString(options: RuntimeToolOptions): string {
        return options.map(env => `${env.name}=${env.value}`).join('\n');
    }

    private stringToOptions(options: string): RuntimeToolOptions {
        return options
            .split('\n')
            .map(env => {
                const arr = env.split('=');
                if (arr[0]) {
                    return {
                        name: arr[0],
                        value: arr[1],
                    };
                } else {
                    return false;
                }
            })
            .filter(Boolean) as RuntimeToolOptions;
    }

    private getEnvOverrides(): RuntimeToolOptions {
        return this.stringToOptions(this.envVarsInput.val() as string);
    }

    private selectOverrideFromFave(event) {
        const elem = $(event.target).parent();
        const name = elem.data('ov-name');
        const optionsStr = elem.data('ov-options');
        const options = this.stringToOptions(optionsStr);

        const tool = this.possibleTools.find(ov => ov.name === name);
        if (tool) {
            const configuredTools = this.loadStateFromUI();
            let configuredTool = configuredTools.find(t => t.name === name);
            if (!configuredTool) {
                configuredTool = {
                    name: name,
                    options: [],
                };
                configuredTools.push(configuredTool);
            }

            configuredTool.options = options;

            this.loadStateIntoUI(configuredTools);
        }
    }

    private newFavoriteOverrideDiv(fave: FavRuntimeTool) {
        const div = $('#overrides-favorite-tpl').children().clone();
        const prefix = fave.name + ': ';
        div.find('.overrides-name').html(prefix + fave.options.replace(/\n/g, ', '));
        div.data('ov-name', fave.name);
        div.data('ov-options', fave.options);
        div.on('click', this.selectOverrideFromFave.bind(this));
        return div;
    }

    private loadFavoritesIntoUI() {
        const favoritesDiv = this.popupDomRoot.find('.runtimetools-favorites');
        favoritesDiv.html('');

        const faves = this.getFavorites();
        for (const fave of faves) {
            const div: any = this.newFavoriteOverrideDiv(fave);
            favoritesDiv.append(div);
        }
    }

    private addToFavorites(override: ConfiguredRuntimeTool) {
        if (override.name === RuntimeToolType.env) return;

        const faves = this.getFavorites();

        const fave: FavRuntimeTool = {
            name: override.name,
            options: this.optionsToString(override.options),
            meta: this.compiler?.baseName || this.compiler?.groupName || this.compiler?.name || this.compiler?.id || '',
        };

        faves.push(fave);

        this.setFavorites(faves);
    }

    private removeFromFavorites(override: ConfiguredRuntimeTool) {
        if (override.name === RuntimeToolType.env) return;

        const overrideOptions = this.optionsToString(override.options);

        const faves = this.getFavorites();
        const faveIdx = faves.findIndex(f => f.name === override.name && f.options === overrideOptions);
        if (faveIdx !== -1) {
            faves.splice(faveIdx, 1);
            this.setFavorites(faves);
        }
    }

    private isAFavorite(override: ConfiguredRuntimeTool) {
        if (override.name === RuntimeToolType.env) return false;

        const overrideOptions = this.optionsToString(override.options);

        const faves = this.getFavorites();
        const fave = faves.find(f => f.name === override.name && f.options === overrideOptions);
        return !!fave;
    }

    private cap(text: string) {
        if (text.length > 0) {
            return text[0].toUpperCase() + text.substring(1);
        }

        return '';
    }

    private loadStateIntoUI(configured: ConfiguredRuntimeTools) {
        this.envVarsInput.val('');

        for (const config of configured) {
            if (config.name === RuntimeToolType.env) {
                this.envVarsInput.val(this.optionsToString(config.options));
            }
        }

        const container = this.popupDomRoot.find('.possible-runtimetools');
        container.html('');

        this.possibleTools = this.compiler?.possibleRuntimeTools || [];

        for (const possibleTool of this.possibleTools) {
            const card = $('#possible-runtime-tool').children().clone();
            card.find('.tool-name').html(this.cap(possibleTool.name));
            card.find('.tool-description').html(possibleTool.description);

            const toolOptionsDiv = card.find('.tool-options');

            const faveButton = card.find('.tool-fav-button');
            faveButton.hide();
            const faveStar = faveButton.find('.tool-fav-btn-icon');

            const config = configured.find(c => c.name === possibleTool.name);

            for (const toolOption of possibleTool.possibleOptions) {
                const optionDiv = $('#possible-runtime-tool-option').children().clone();
                optionDiv.attr('name', toolOption.name);
                const display_text = this.cap(toolOption.name);
                optionDiv.find('.tool-option-name').html(display_text);

                const select = optionDiv.find('select');
                select.data('tool-name', possibleTool.name);
                select.data('tool-option', toolOption.name);

                const option = $('<option />');
                option.html('');
                option.val('');
                select.append(option);

                for (const toolOptionValue of toolOption.possibleValues) {
                    const option = $('<option />');
                    option.html(toolOptionValue);
                    option.val(toolOptionValue);

                    if (config) {
                        const found = config.options.find(
                            configuredOption =>
                                configuredOption.name === toolOption.name && configuredOption.value === toolOptionValue,
                        );
                        if (found) option.attr('selected', 'selected');
                    }

                    select.append(option);
                }

                select.off('change').on('change', () => {
                    const name = possibleTool.name;
                    assert(name !== RuntimeToolType.env);
                    const configured = this.loadStateFromUI();
                    const configuredTool = configured.find(tool => tool.name === name);

                    if (configuredTool) {
                        if (this.isAFavorite(configuredTool)) {
                            faveStar.removeClass('far').addClass('fas');
                        } else {
                            faveStar.removeClass('fas').addClass('far');
                        }

                        if (configuredTool.options.length !== 0) {
                            faveButton.show();
                        } else {
                            faveButton.hide();
                        }
                    } else {
                        faveStar.removeClass('fas').addClass('far');
                    }
                });

                toolOptionsDiv.append(optionDiv);
            }

            if (config && this.isAFavorite(config)) {
                faveStar.removeClass('far').addClass('fas');
            }
            faveButton.show();

            faveButton.on('click', () => {
                const name = possibleTool.name;
                assert(name !== RuntimeToolType.env);

                const configured = this.loadStateFromUI();
                const configuredTool = configured.find(tool => tool.name === name);
                if (configuredTool) {
                    if (this.isAFavorite(configuredTool)) {
                        this.removeFromFavorites(configuredTool);
                        faveStar.removeClass('fas').addClass('far');
                    } else {
                        this.addToFavorites(configuredTool);
                        faveStar.removeClass('far').addClass('fas');
                    }
                }

                this.loadFavoritesIntoUI();
            });

            container.append(card);
        }

        this.loadFavoritesIntoUI();
    }

    set(configured: ConfiguredRuntimeTools) {
        this.configured = configured;
        this.updateButton();
    }

    setDefaults() {
        this.configured = [];

        this.updateButton();
    }

    setCompiler(compilerId: string, languageId?: string) {
        this.compiler = options.compilers.find(c => c.id === compilerId);
    }

    get(): ConfiguredRuntimeTools | undefined {
        if (this.compiler) {
            return this.configured;
        } else {
            return undefined;
        }
    }

    private getFavorites(): FavRuntimeTools {
        return JSON.parse(localStorage.get(FAV_RUNTIMETOOLS_STORE_KEY, '[]'));
    }

    private setFavorites(faves: FavRuntimeTools) {
        localStorage.set(FAV_RUNTIMETOOLS_STORE_KEY, JSON.stringify(faves));
    }

    private updateButton() {
        const selected = this.get();
        if (selected && selected.length > 0) {
            this.dropdownButton
                .addClass('btn-success')
                .removeClass('btn-light')
                .prop(
                    'title',
                    'Current tools:\n' +
                        selected
                            .map(ov => {
                                return '- ' + ov.name;
                            })
                            .join('\n'),
                );
        } else {
            this.dropdownButton.removeClass('btn-success').addClass('btn-light').prop('title', 'Overrides');
        }
    }

    show() {
        this.loadStateIntoUI(this.configured);

        const lastOverrides = JSON.stringify(this.configured);

        const popup = this.popupDomRoot.modal();
        // popup is shared, so clear the events first
        popup.off('hidden.bs.modal').on('hidden.bs.modal', () => {
            this.configured = this.loadStateFromUI();

            const newOverrides = JSON.stringify(this.configured);

            if (lastOverrides !== newOverrides) {
                this.updateButton();
                this.onChangeCallback();
            }
        });
    }
}
