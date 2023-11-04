import {BaseCompiler} from '../base-compiler.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {TableGenParser} from './argument-parsers.js';
import {CompilerOverrideType} from '../../types/compilation/compiler-overrides.interfaces.js';

export class TableGenCompiler extends BaseCompiler {
    static get key() {
        return 'tablegen';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        const options: string[] = ['-o', outputFilename];
        if (this.compiler.includePath) {
            options.push(`-I${this.compiler.includePath}`);
        }
        return options;
    }

    override isCfgCompiler() {
        return false;
    }

    override getArgumentParser() {
        return TableGenParser;
    }

    override async populatePossibleOverrides() {
        const possibleActions = await TableGenParser.getPossibleActions(this);
        if (possibleActions.length > 0) {
            this.compiler.possibleOverrides?.push({
                name: CompilerOverrideType.action,
                display_title: 'Action',
                description:
                    'The action to perform, which is the backend you wish to ' +
                    'run. By default, the records are just printed as text.',
                flags: ['<value>'],
                values: possibleActions,
                default: '--print-records',
            });
        }

        await super.populatePossibleOverrides();
    }
}
