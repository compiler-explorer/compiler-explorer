import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseObjdumper} from './base.js';

export class PlainObjdumper extends BaseObjdumper {
    constructor() {
        super([], []);
    }

    override getArgs(
        outputFilename: string,
        demangle?: boolean,
        intelAsm?: boolean,
        staticReloc?: boolean,
        dynamicReloc?: boolean,
        objdumperArguments?: string[],
        filters?: ParseFiltersAndOutputOptions,
    ) {
        const args: string[] = [];
        if (objdumperArguments) args.push(...objdumperArguments);
        args.push(outputFilename);
        return args;
    }

    static override get key() {
        return 'plain';
    }
}
