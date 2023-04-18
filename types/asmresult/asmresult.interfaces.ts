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
    labelDefinitions?: Record<string, number>;
    parsingTime?: string;
    filteredCount?: number;
    externalParserUsed?: boolean;
    // TODO(#4655) A few compilers seem to assign strings here. It might be ok but we should look into it more.
    objdumpTime?: number | string;
    execTime?: string;
    languageId?: string;
};

export type IRResultLine = ParsedAsmResultLine & {
    scope?: string;
};
