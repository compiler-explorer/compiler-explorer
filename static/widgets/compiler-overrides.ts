import $ from 'jquery';
import {
    CompilerOverrideType,
    ConfiguredOverrides,
    EnvvarOverrides,
} from '../../types/compilation/compiler-overrides.interfaces.js';

export type CompilerOverridesChangeCallback = () => void;

export class CompilerOverridesWidget {
    private domRoot: JQuery;
    private popupDomRoot: JQuery<HTMLElement>;
    private envvarsInput: JQuery<HTMLElement>;
    private dropdownButton: JQuery;
    private onChangeCallback: CompilerOverridesChangeCallback;
    private configured: ConfiguredOverrides = [];

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
    }

    set(configured: ConfiguredOverrides) {
        this.configured = configured;
        this.updateButton();
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
