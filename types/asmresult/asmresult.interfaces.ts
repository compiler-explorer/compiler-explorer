import { ResultLineTag } from '../resultline/resultline.interfaces';

export class ParsedAsmResult {
    asm: ResultLineTag[];
    labelDefinitions: Map<string, number>;
    parsingTime: string;
    filteredCount: number;
    externalParserUsed?: boolean;
}
