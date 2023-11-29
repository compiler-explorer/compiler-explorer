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
import {
    CompilerOverrideType,
    ConfiguredOverride,
    ConfiguredOverrides,
    EnvVarOverrides,
} from '../../types/compilation/compiler-overrides.interfaces.js';
import {options} from '../options.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {assert, unwrap} from '../assert.js';
import {localStorage} from '../local.js';

const FAV_OVERRIDES_STORE_KEY = 'favoverrides';

export type CompilerOverridesChangeCallback = () => void;

type FavOverride = {
    name: CompilerOverrideType;
    value: string;
    meta: string;
};

type FavOverrides = FavOverride[];

class IncompatibleState {
    constructor(reason: string) {
        this.reason = reason;
    }
    reason: string;
}
class InactiveState {}
class ActiveState {}
type OverrideState = IncompatibleState | InactiveState | ActiveState;

export class CompilerOverridesWidget {
    private domRoot: JQuery;
    private popupDomRoot: JQuery<HTMLElement>;
    private envVarsInput: JQuery<HTMLElement>;
    private dropdownButton: JQuery;
    private onChangeCallback: CompilerOverridesChangeCallback;
    private configured: ConfiguredOverrides = [];
    private compiler: CompilerInfo | undefined;

    constructor(domRoot: JQuery, dropdownButton: JQuery, onChangeCallback: CompilerOverridesChangeCallback) {
        this.domRoot = domRoot;
        this.popupDomRoot = $('#overrides-selection');
        this.dropdownButton = dropdownButton;
        this.envVarsInput = this.popupDomRoot.find('.envvars');
        this.onChangeCallback = onChangeCallback;
    }

    private loadStateFromUI(): ConfiguredOverrides {
        const overrides: ConfiguredOverrides = [];

        const envOverrides = this.getEnvOverrides();
        if (envOverrides.length > 0) {
            overrides.push({
                name: CompilerOverrideType.env,
                values: envOverrides,
            });
        }

        const selects = this.popupDomRoot.find('select');
        for (const select of selects) {
            const jqSelect = $(select);
            const rawName = jqSelect.attr('name');
            const val = jqSelect.val();
            if (val) {
                const name = rawName as CompilerOverrideType;
                assert(name !== CompilerOverrideType.env);
                overrides.push({
                    name: name,
                    value: val.toString(),
                });
            }
        }

        return overrides;
    }

    private envvarsToString(envVars: EnvVarOverrides): string {
        return envVars.map(env => `${env.name}=${env.value}`).join('\n');
    }

    private stringToEnvvars(envVars: string): EnvVarOverrides {
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
            .filter(Boolean) as EnvVarOverrides;
    }

    private getEnvOverrides(): EnvVarOverrides {
        return this.stringToEnvvars(this.envVarsInput.val() as string);
    }

    private selectOverrideFromFave(event) {
        const elem = $(event.target).parent();
        const name = elem.data('ov-name');
        const value = elem.data('ov-value');

        const possibleOverride = this.compiler?.possibleOverrides?.find(ov => ov.name === name);
        if (possibleOverride) {
            const override = possibleOverride.values.find(v => v.value === value);
            if (override) {
                const currentOverrides = this.loadStateFromUI();
                const configOv = currentOverrides.find(ov => ov.name === name);
                if (configOv) {
                    assert(configOv.name !== CompilerOverrideType.env);
                    // If it is already enabled, clear the value.
                    configOv.value = configOv.value === value ? '' : value;
                } else {
                    currentOverrides.push({
                        name: name,
                        value: value,
                    });
                }

                this.loadStateIntoUI(currentOverrides);
            }
        }
    }

    private newFavoriteOverrideDiv(fave: FavOverride, state: OverrideState) {
        const div = $('#overrides-favorite-tpl').children().clone();
        const prefix = fave.name + ': ';
        const btn = div.find('.overrides-name');
        btn.html(prefix + fave.value);
        if (state instanceof ActiveState) {
            btn.addClass('active');
        } else if (state instanceof IncompatibleState) {
            btn.prop('disabled', true);
            btn.prop('data-toggle', 'tooltip');
            btn.prop('data-placement', 'top');
            btn.prop('title', state.reason);
        }
        div.data('ov-name', fave.name);
        div.data('ov-value', fave.value);
        div.on('click', this.selectOverrideFromFave.bind(this));
        return div;
    }

    private loadFavoritesIntoUI() {
        const favoritesDiv = this.popupDomRoot.find('.overrides-favorites');
        favoritesDiv.html('');

        const faves = this.getFavorites();
        const current_overrides = this.get();

        for (const fave of faves) {
            let state: OverrideState = new IncompatibleState(
                'This override is not compatible with the current compiler.',
            );
            const possible = this.compiler?.possibleOverrides?.find(ov => ov.name === fave.name);
            if (possible) {
                state = new InactiveState();
                if (!possible.values.find(ov => ov.value === fave.value)) {
                    state = new IncompatibleState(
                        'The value of this override is not compatible with the current compiler.',
                    );
                } else if (
                    current_overrides?.find(ov => {
                        return ov.name !== CompilerOverrideType.env && ov.name === fave.name && ov.value === fave.value;
                    })
                ) {
                    state = new ActiveState();
                }
            }

            const div: any = this.newFavoriteOverrideDiv(fave, state);
            favoritesDiv.append(div);
        }
    }

    private addToFavorites(override: ConfiguredOverride) {
        if (override.name === CompilerOverrideType.env || !override.value) return;

        const faves = this.getFavorites();

        const fave: FavOverride = {
            name: override.name,
            value: override.value,
            meta: this.compiler?.baseName || this.compiler?.groupName || this.compiler?.name || this.compiler?.id || '',
        };

        faves.push(fave);

        this.setFavorites(faves);
    }

    private removeFromFavorites(override: ConfiguredOverride) {
        if (override.name === CompilerOverrideType.env || !override.value) return;

        const faves = this.getFavorites();
        const faveIdx = faves.findIndex(f => f.name === override.name && f.value === override.value);
        if (faveIdx !== -1) {
            faves.splice(faveIdx, 1);
            this.setFavorites(faves);
        }
    }

    private isAFavorite(override: ConfiguredOverride) {
        if (override.name === CompilerOverrideType.env || !override.value) return false;

        const faves = this.getFavorites();
        const fave = faves.find(f => f.name === override.name && f.value === override.value);
        return !!fave;
    }

    private loadStateIntoUI(configured: ConfiguredOverrides) {
        this.envVarsInput.val('');

        for (const config of configured) {
            if (config.name === CompilerOverrideType.env) {
                this.envVarsInput.val(this.envvarsToString(config.values));
            }
        }

        const container = this.popupDomRoot.find('.possible-overrides');
        container.html('');
        if (this.compiler && this.compiler.possibleOverrides) {
            for (const possibleOverride of this.compiler.possibleOverrides) {
                const card = $('#possible-override').children().clone();
                card.find('.override-name').html(possibleOverride.display_title);
                card.find('.override-description').html(possibleOverride.description);

                const select = card.find<HTMLSelectElement>('.override select');
                select.attr('name', possibleOverride.name);

                const faveButton = card.find('.override-fav-button');
                const faveStar = faveButton.find('.override-fav-btn-icon');
                faveButton.hide();

                const config = configured.find(c => c.name === possibleOverride.name);

                let option = $('<option />');
                select.append(option);

                for (const value of possibleOverride.values) {
                    option = $('<option />');
                    option.html(value.name);
                    option.val(value.value);

                    if (
                        config &&
                        config.name !== CompilerOverrideType.env &&
                        config.value &&
                        config.value === value.value
                    ) {
                        option.attr('selected', 'selected');

                        if (this.isAFavorite(config)) {
                            faveStar.removeClass('far').addClass('fas');
                        }

                        faveButton.show();
                    }

                    select.append(option);
                }

                select.off('change').on('change', () => {
                    const option = select.find('option:selected');
                    if (option.length > 0) {
                        const value = unwrap(option.val()).toString();
                        const name = possibleOverride.name;
                        assert(name !== CompilerOverrideType.env);

                        const ov: ConfiguredOverride = {
                            name: name,
                            value: value,
                        };

                        if (this.isAFavorite(ov)) {
                            faveStar.removeClass('far').addClass('fas');
                        } else {
                            faveStar.removeClass('fas').addClass('far');
                        }

                        if (ov.value !== '') {
                            faveButton.show();
                        } else {
                            faveButton.hide();
                        }
                    }

                    this.configured = this.loadStateFromUI();
                    this.loadFavoritesIntoUI();
                });

                faveButton.on('click', () => {
                    const option = select.find('option:selected');
                    if (option.length > 0) {
                        const value = unwrap(option.val()).toString();
                        const name = possibleOverride.name;
                        assert(name !== CompilerOverrideType.env);

                        const ov: ConfiguredOverride = {name, value};
                        if (this.isAFavorite(ov)) {
                            this.removeFromFavorites(ov);
                            faveStar.removeClass('fas').addClass('far');
                        } else {
                            this.addToFavorites(ov);
                            faveStar.removeClass('far').addClass('fas');
                        }
                    }

                    this.loadFavoritesIntoUI();
                });

                container.append(card);
            }
        }

        this.configured = configured;
        this.loadFavoritesIntoUI();
    }

    set(configured: ConfiguredOverrides) {
        this.configured = configured;
        this.updateButton();
    }

    setDefaults() {
        this.configured = [];

        if (this.compiler && this.compiler.possibleOverrides) {
            for (const ov of this.compiler.possibleOverrides) {
                if (ov.name !== CompilerOverrideType.env && ov.default) {
                    this.configured.push({
                        name: ov.name,
                        value: ov.default,
                    });
                }
            }
        }

        this.updateButton();
    }

    setCompiler(compilerId: string, languageId?: string) {
        this.compiler = options.compilers.find(c => c.id === compilerId);
    }

    get(): ConfiguredOverrides | undefined {
        if (this.compiler) {
            return this.configured;
        } else {
            return undefined;
        }
    }

    private getFavorites(): FavOverrides {
        return JSON.parse(localStorage.get(FAV_OVERRIDES_STORE_KEY, '[]'));
    }

    private setFavorites(faves: FavOverrides) {
        localStorage.set(FAV_OVERRIDES_STORE_KEY, JSON.stringify(faves));
    }

    private updateButton() {
        const selected = this.get();
        if (selected && selected.length > 0) {
            this.dropdownButton
                .addClass('btn-success')
                .removeClass('btn-light')
                .prop(
                    'title',
                    'Current overrides:\n' +
                        selected
                            .map(ov => {
                                let line = '- ' + ov.name;
                                if (ov.name !== CompilerOverrideType.env && ov.value) {
                                    line += ' = ' + ov.value;
                                }
                                return line;
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
