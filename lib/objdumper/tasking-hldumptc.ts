import {BaseObjdumper} from './base';

export class TaskingHlObjdumper extends BaseObjdumper {
    static override get key() {
        return 'tasking-hldumptc';
    }

    override getDefaultArgs(
        outputFilename: string,
        demangle?: boolean,
        intelAsm?: boolean,
        staticReloc?: boolean,
        dynamicReloc?: boolean,
    ) {
        return [
            '-F2', // dump format: only show section-dump.
            '-cc', // dump class: only dump code
            '-Sn', // dump symbol: none
            outputFilename,
        ];
    }
}
