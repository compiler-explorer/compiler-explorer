export type ResultLineTag = {
    line?: number;
    column?: number;
    file?: string;
    text: string;
    severity: number;
    endline?: number;
    endcolumn?: number;
};

export type ResultLine = {
    text: string;
    tag?: ResultLineTag;
};
