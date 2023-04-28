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
