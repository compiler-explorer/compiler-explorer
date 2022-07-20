export type Link = {
    text: string;
    url: string;
};

export type MessageWithLocation = {
    line?: number;
    column?: number;
    file?: string;
    text: string;
    endline?: number;
    endcolumn?: number;
};

export type ResultLineTag = MessageWithLocation & {
    severity: number;
    link?: Link;
    flow?: MessageWithLocation[];
};

export type ResultLine = {
    text: string;
    tag?: ResultLineTag;
};
