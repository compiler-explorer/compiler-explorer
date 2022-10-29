export type AsmResultSource = {
    file: string | null;
    line?: number;
    column?: number;
    mainsource?: boolean;
};

export type AsmResultLink = {
    offset: number;
    length: number;
    to: number;
};

export type AsmResultLabelRange = {
    startCol: number;
    endCol: number;
};

export type AsmResultLabel = {
    name: string;
    range: AsmResultLabelRange;
};

export type ParsedAsmResultLine = {
    text: string;
    opcodes?: string[];
    address?: number;
    disassembly?: string;
    source?: AsmResultSource | null;
    links?: AsmResultLink[];
    labels?: AsmResultLabel[];
};

export type ParsedAsmResult = {
    asm: ParsedAsmResultLine[];
    labelDefinitions?: Map<string, number>;
    parsingTime?: string;
    filteredCount?: number;
    externalParserUsed?: boolean;
    objdumpTime?: number;
    execTime?: string;
    languageId?: string;
};
