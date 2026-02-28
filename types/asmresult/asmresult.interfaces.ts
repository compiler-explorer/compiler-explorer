export type AsmResultSource = {
    file: string | null;
    line: number | null;
    column?: number;
    mainsource?: boolean;
};

export type AsmResultLabelRange = {
    startCol: number;
    endCol: number;
};

export type AsmResultLabel = {
    name: string;
    target?: string;
    range: AsmResultLabelRange;
};

export type ParsedAsmResultLine = {
    text: string;
    opcodes?: string[];
    address?: number;
    disassembly?: string;
    source?: AsmResultSource | null;
    labels?: AsmResultLabel[];
};

export type ParsedAsmResult = {
    asm: ParsedAsmResultLine[];
    labelDefinitions?: Record<string, number>;
    parsingTime?: number;
    filteredCount?: number;
    externalParserUsed?: boolean;
    objdumpTime?: number;
    execTime?: number;
    languageId?: string;
    asmKeywordTypes?: string[];
};

export type IRResultLine = ParsedAsmResultLine & {
    scope?: string;
};
