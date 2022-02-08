import { ParseFilters } from '../../types/features/filters.interfaces';
import { ResultLineTag } from '../../types/resultline/resultline.interfaces';

export class ParsedAsmResult {
    asm: IResultLineTag[];
    labelDefinitions: Map<string, number>;
    parsingTime: string;
    filteredCount: number;
}

export interface IExternalParser {
    objdumpAndParseAssembly(buildfolder: string, objdumpArgs: string[],
        filters: ParseFilters): Promise<ParsedAsmResult>;
    parseAssembly(filepath: string, filters: ParseFilters): Promise<ParsedAsmResult>;
}
