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
    ConfiguredOverrides,
    EnvvarOverrides,
} from '../../types/compilation/compiler-overrides.interfaces.js';
import {options} from '../options.js';
import {CompilerInfo} from '../compiler.interfaces.js';

export type CompilerOverridesChangeCallback = () => void;

export class CompilerOverridesWidget {
    private domRoot: JQuery;
    private popupDomRoot: JQuery<HTMLElement>;
    private envvarsInput: JQuery<HTMLElement>;
    private dropdownButton: JQuery;
    private onChangeCallback: CompilerOverridesChangeCallback;
    private configured: ConfiguredOverrides = [];
    private compiler: CompilerInfo | undefined;

    constructor(domRoot: JQuery, dropdownButton: JQuery, onChangeCallback: CompilerOverridesChangeCallback) {
        this.domRoot = domRoot;
        this.popupDomRoot = $('#overrides-selection');
        this.dropdownButton = dropdownButton;
        this.envvarsInput = this.popupDomRoot.find('.envvars');
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
            const name = jqSelect.attr('name');
            const val = jqSelect.val();
            if (val) {
                overrides.push({
                    name: name as CompilerOverrideType,
                    value: val.toString(),
                });
            }
        }

        return overrides;
    }

    private envvarsToString(envvars: EnvvarOverrides): string {
        return envvars.map(env => `${env.name}=${env.value}`).join('\n');
    }

    private stringToEnvvars(envvars: string): EnvvarOverrides {
        return envvars
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
            .filter(Boolean) as EnvvarOverrides;
    }

    private getEnvOverrides(): EnvvarOverrides {
        return this.stringToEnvvars(this.envvarsInput.val() as string);
    }

    private loadStateIntoUI(configured: ConfiguredOverrides) {
        this.envvarsInput.val('');

        for (const config of configured) {
            if (config.name === CompilerOverrideType.env) {
                this.envvarsInput.val(this.envvarsToString(config.values || []));
            }
        }

        if (this.compiler && this.compiler.possibleOverrides) {
            const container = this.popupDomRoot.find('.possible-overrides');
            container.html('');
            for (const possibleOverride of this.compiler.possibleOverrides) {
                const card = $(
                    '<div class="card">' +
                        '<div class="card-header">' +
                        '<span class="override-name"></span>' +
                        '<span class="override">' +
                        '<select class="custom-select custom-select-sm"></select>' +
                        '</span>' +
                        '</div>' +
                        '<div class="card-body">' +
                        '<span class="override-description"></span>' +
                        '</div>' +
                        '</div>',
                );
                card.find('.override-name').html(possibleOverride.display_title);
                card.find('.override-description').html(possibleOverride.description);

                const select = card.find<HTMLSelectElement>('.override select');
                select.attr('name', possibleOverride.name);

                const config = configured.find(c => c.name === possibleOverride.name);

                let option = $('<option />');
                select.append(option);

                for (const value of possibleOverride.values) {
                    option = $('<option />');
                    option.html(value.name);
                    option.val(value.value);

                    if (config && config.value && config.value === value.value) {
                        option.attr('selected', 'selected');
                    }

                    select.append(option);
                }

                container.append(card);
            }
        }
    }

    set(configured: ConfiguredOverrides) {
        this.configured = configured;
        this.updateButton();
    }

    setCompiler(compilerId: string, languageId?: string) {
        this.compiler = options.compilers.find(c => c.id === compilerId);
    }

    get(): ConfiguredOverrides {
        return this.configured;
    }

    private updateButton() {
        const selected = this.get();
        if (selected.length > 0) {
            this.dropdownButton
                .addClass('btn-success')
                .removeClass('btn-light')
                .prop('title', 'Current overrides:\n' + selected.map(ov => '- ' + ov.name).join('\n'));
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
