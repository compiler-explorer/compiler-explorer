export type Link = {
    text: string;
    url: string;
};

export type ResultLineTag = {
    line?: number;
    column?: number;
    file?: string;
    text: string;
    severity: number;
    endline?: number;
    endcolumn?: number;
    link?: Link;
};

export type ResultLine = {
    text: string;
    tag?: ResultLineTag;
};
