import {TricoreGNUObjdumper} from './tricoregnu';

export class TricoreTaskingObjdumper extends TricoreGNUObjdumper {
    static override get key() {
        return 'tricore-tasking';
    }

    override getDefaultArgs(
        outputFilename: string,
        demangle?: boolean,
        intelAsm?: boolean,
        staticReloc?: boolean,
        dynamicReloc?: boolean,
    ) {
        const args = ['-mtricore', '--prefix-addresses', '-d', outputFilename, '-l', ...this.widthOptions];

        if (staticReloc) args.push('-r');
        // if (dynamicReloc) args.push('-R');
        if (demangle) args.push('-C');
        if (intelAsm) args.push(...this.intelAsmOptions);

        return args;
    }
}
