export type ResultLineTag = {
    line?: number;
    column?: number;
    file?: string;
    text: string;
};

export type ResultLine = {
    text: string;
    tag?: ResultLineTag;
};
