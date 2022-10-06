import {ResultLineTag} from '../resultline/resultline.interfaces';

export type ParsedAsmResult = {
    asm: ResultLineTag[];
    labelDefinitions: Map<string, number>;
    parsingTime: string;
    filteredCount: number;
    externalParserUsed?: boolean;
    objdumpTime?: number;
    execTime?: string;
    languageId?: string;
};
