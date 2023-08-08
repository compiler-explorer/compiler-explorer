import type {ParsedAsmResult} from '../../types/asmresult/asmresult.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

export interface IExternalParser {
    objdumpAndParseAssembly(
        buildfolder: string,
        objdumpArgs: string[],
        filters: ParseFiltersAndOutputOptions,
    ): Promise<ParsedAsmResult>;
    parseAssembly(filepath: string, filters: ParseFiltersAndOutputOptions): Promise<ParsedAsmResult>;
}
