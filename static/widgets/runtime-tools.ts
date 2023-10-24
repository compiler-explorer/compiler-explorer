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
    RuntimeToolOption,
    RuntimeToolOptions,
    RuntimeToolType,
} from '../../types/execution/execution.interfaces.js';

const FAV_RUNTIMETOOLS_STORE_KEY = 'favruntimetools';

export type RuntimeToolsChangeCallback = () => void;

type FavRuntimeTool = {
    name: RuntimeToolType;
    options: RuntimeToolOption[];
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

    constructor(domRoot: JQuery, dropdownButton: JQuery, onChangeCallback: RuntimeToolsChangeCallback) {
        this.domRoot = domRoot;
        this.popupDomRoot = $('#runtimetools-selection');
        this.dropdownButton = dropdownButton;
        this.envVarsInput = this.popupDomRoot.find('.envvars');
        this.onChangeCallback = onChangeCallback;
    }

    private loadStateFromUI(): ConfiguredRuntimeTools {
        const overrides: ConfiguredRuntimeTools = [];

        const envOverrides = this.getEnvOverrides();
        if (envOverrides.length > 0) {
            overrides.push({
                name: RuntimeToolType.env,
                options: envOverrides,
            });
        }

        const selects = this.popupDomRoot.find('select');
        for (const select of selects) {
            const jqSelect = $(select);
            const rawName = jqSelect.attr('name');
            const val = jqSelect.val();
            if (val) {
                const name = rawName as RuntimeToolType;
                assert(name !== RuntimeToolType.env);
                overrides.push({
                    name: name,
                    options: [],
                });
            }
        }

        return overrides;
    }

    private envvarsToString(envVars: RuntimeToolOptions): string {
        return envVars.map(env => `${env.name}=${env.value}`).join('\n');
    }

    private stringToEnvvars(envVars: string): RuntimeToolOptions {
        return envVars
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
        return this.stringToEnvvars(this.envVarsInput.val() as string);
    }

    private selectOverrideFromFave(event) {
        // const elem = $(event.target).parent();
        // const name = elem.data('ov-name');
        // const value = elem.data('ov-value');
        // const possibleOverride = this.compiler?.possibleOverrides?.find(ov => ov.name === name);
        // if (possibleOverride) {
        //     const override = possibleOverride.values.find(v => v.value === value);
        //     if (override) {
        //         const currentOverrides = this.loadStateFromUI();
        //         const configOv = currentOverrides.find(ov => ov.name === name);
        //         if (configOv) {
        //             assert(configOv.name !== RuntimeToolType.env);
        //             configOv.options = value;
        //         } else {
        //             currentOverrides.push({
        //                 name: name,
        //                 options: value,
        //             });
        //         }
        //         this.loadStateIntoUI(currentOverrides);
        //     }
        // }
    }

    private newFavoriteOverrideDiv(fave: FavRuntimeTool) {
        const div = $('#overrides-favorite-tpl').children().clone();
        const prefix = fave.name + ': ';
        // div.find('.overrides-name').html(prefix + fave.options);
        div.data('ov-name', fave.name);
        // div.data('ov-value', fave.value);
        div.on('click', this.selectOverrideFromFave.bind(this));
        return div;
    }

    private loadFavoritesIntoUI() {
        const favoritesDiv = this.popupDomRoot.find('.overrides-favorites');
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
            options: override.options,
            meta: this.compiler?.baseName || this.compiler?.groupName || this.compiler?.name || this.compiler?.id || '',
        };

        faves.push(fave);

        this.setFavorites(faves);
    }

    private removeFromFavorites(override: ConfiguredRuntimeTool) {
        if (override.name === RuntimeToolType.env) return;

        const faves = this.getFavorites();
        const faveIdx = faves.findIndex(f => f.name === override.name && f.options === override.options);
        if (faveIdx !== -1) {
            faves.splice(faveIdx, 1);
            this.setFavorites(faves);
        }
    }

    private isAFavorite(override: ConfiguredRuntimeTool) {
        if (override.name === RuntimeToolType.env) return false;

        const faves = this.getFavorites();
        const fave = faves.find(f => f.name === override.name && f.options === override.options);
        return !!fave;
    }

    private loadStateIntoUI(configured: ConfiguredRuntimeTools) {
        this.envVarsInput.val('');

        for (const config of configured) {
            if (config.name === RuntimeToolType.env) {
                this.envVarsInput.val(this.envvarsToString(config.options));
            }
        }

        // const container = this.popupDomRoot.find('.possible-overrides');
        // container.html('');
        // if (this.compiler && this.compiler.possibleOverrides) {
        //     for (const possibleOverride of this.compiler.possibleOverrides) {
        //         const card = $('#possible-override').children().clone();
        //         card.find('.override-name').html(possibleOverride.display_title);
        //         card.find('.override-description').html(possibleOverride.description);

        //         const select = card.find<HTMLSelectElement>('.override select');
        //         select.attr('name', possibleOverride.name);

        //         const faveButton = card.find('.override-fav-button');
        //         const faveStar = faveButton.find('.override-fav-btn-icon');
        //         faveButton.hide();

        //         const config = configured.find(c => c.name === possibleOverride.name);

        //         let option = $('<option />');
        //         select.append(option);

        //         for (const value of possibleOverride.values) {
        //             option = $('<option />');
        //             option.html(value.name);
        //             option.val(value.value);

        //             if (
        //                 config &&
        //                 config.name !== CompilerOverrideType.env &&
        //                 config.value &&
        //                 config.value === value.value
        //             ) {
        //                 option.attr('selected', 'selected');

        //                 if (this.isAFavorite(config)) {
        //                     faveStar.removeClass('far').addClass('fas');
        //                 }

        //                 faveButton.show();
        //             }

        //             select.append(option);
        //         }

        //         select.off('change').on('change', () => {
        //             const option = select.find('option:selected');
        //             if (option.length > 0) {
        //                 const value = unwrap(option.val()).toString();
        //                 const name = possibleOverride.name;
        //                 assert(name !== CompilerOverrideType.env);

        //                 const ov: ConfiguredOverride = {
        //                     name: name,
        //                     value: value,
        //                 };

        //                 if (this.isAFavorite(ov)) {
        //                     faveStar.removeClass('far').addClass('fas');
        //                 } else {
        //                     faveStar.removeClass('fas').addClass('far');
        //                 }

        //                 if (ov.value !== '') {
        //                     faveButton.show();
        //                 } else {
        //                     faveButton.hide();
        //                 }
        //             }
        //         });

        //         faveButton.on('click', () => {
        //             const option = select.find('option:selected');
        //             if (option.length > 0) {
        //                 const value = unwrap(option.val()).toString();
        //                 const name = possibleOverride.name;
        //                 assert(name !== CompilerOverrideType.env);

        //                 const ov: ConfiguredOverride = {name, value};
        //                 if (this.isAFavorite(ov)) {
        //                     this.removeFromFavorites(ov);
        //                     faveStar.removeClass('fas').addClass('far');
        //                 } else {
        //                     this.addToFavorites(ov);
        //                     faveStar.removeClass('far').addClass('fas');
        //                 }
        //             }

        //             this.loadFavoritesIntoUI();
        //         });

        //         container.append(card);
        //     }
        // }

        this.loadFavoritesIntoUI();
    }

    set(configured: ConfiguredRuntimeTools) {
        this.configured = configured;
        this.updateButton();
    }

    setDefaults() {
        this.configured = [];

        // if (this.compiler && this.compiler.possibleOverrides) {
        //     for (const ov of this.compiler.possibleOverrides) {
        //         if (ov.name !== RuntimeToolType.env && ov.default) {
        //             this.configured.push({
        //                 name: ov.name,
        //                 value: ov.default,
        //             });
        //         }
        //     }
        // }

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
