import {ParsedAsmResult} from '../../types/asmresult/asmresult.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';

export interface IExternalParser {
    objdumpAndParseAssembly(
        buildfolder: string,
        objdumpArgs: string[],
        filters: ParseFiltersAndOutputOptions,
    ): Promise<ParsedAsmResult>;
    parseAssembly(filepath: string, filters: ParseFiltersAndOutputOptions): Promise<ParsedAsmResult>;
}
