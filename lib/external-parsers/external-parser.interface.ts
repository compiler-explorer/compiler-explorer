import { IParseFilters } from '../../types/features/filters.interfaces';
import { IResultLineTag } from '../../types/resultline/resultline.interfaces';

export interface IParsedAsmResult {
    asm: IResultLineTag[];
    labelDefinitions: Set<string, number>;
    parsingTime: string;
    filteredCount: number;
}

export interface IExternalParser {
    objdumpAndParseAssembly(buildfolder: string, objdumpArgs: string[],
        filters: IParseFilters): Promise<IParsedAsmResult>;
    parseAssembly(filepath: string, filters: IParseFilters): Promise<IParsedAsmResult>;
}
