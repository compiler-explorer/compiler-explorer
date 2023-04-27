import $ from 'jquery';
import {
    CompilerOverrideType,
    ConfiguredOverrides,
    EnvvarOverrides,
} from '../../types/compilation/compiler-overrides.interfaces.js';

//const FAV_LIBS_STORE_KEY = 'favlibs';

export type CompilerOverridesChangeCallback = () => void;

export class CompilerOverridesWidget {
    private domRoot: JQuery;
    private modalDomRoot: JQuery<HTMLElement>;
    private envvarsInput: JQuery<HTMLElement>;
    private dropdownButton: JQuery;
    private onChangeCallback: CompilerOverridesChangeCallback;

    constructor(domRoot: JQuery, dropdownButton: JQuery, onChangeCallback: CompilerOverridesChangeCallback) {
        this.domRoot = domRoot;
        this.modalDomRoot = $('#overrides-selection');
        this.dropdownButton = dropdownButton;
        this.envvarsInput = this.modalDomRoot.find('.envvars');
        this.onChangeCallback = onChangeCallback;
    }

    get(): ConfiguredOverrides {
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
        for (const config of configured) {
            if (config.name === CompilerOverrideType.env) {
                this.envvarsInput.val(this.envvarsToString(config.values || []));
            }
        }
    }

    set(configured: ConfiguredOverrides) {
        // put settings into UI
        this.loadStateIntoUI(configured);
        this.updateButton();
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
        const lastOverrides = JSON.stringify(this.get());
        this.modalDomRoot.modal().on('hidden.bs.modal', () => {
            const newOverrides = JSON.stringify(this.get());
            if (lastOverrides !== newOverrides) {
                this.updateButton();
                this.onChangeCallback();
            }
        });
    }
}
