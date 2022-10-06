import {ParsedAsmResult} from '../../types/asmresult/asmresult.interfaces';
import {ParseFilters} from '../../types/features/filters.interfaces';

export interface IExternalParser {
    objdumpAndParseAssembly(
        buildfolder: string,
        objdumpArgs: string[],
        filters: ParseFilters,
    ): Promise<ParsedAsmResult>;
    parseAssembly(filepath: string, filters: ParseFilters): Promise<ParsedAsmResult>;
}
