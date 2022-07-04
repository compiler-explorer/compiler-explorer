import {ParsedAsmResult} from '../../types/asmresult/asmresult.interfaces.js';
import {ParseFilters} from '../../types/features/filters.interfaces.js';

export interface IExternalParser {
    objdumpAndParseAssembly(
        buildfolder: string,
        objdumpArgs: string[],
        filters: ParseFilters,
    ): Promise<ParsedAsmResult>;
    parseAssembly(filepath: string, filters: ParseFilters): Promise<ParsedAsmResult>;
}
