export type Link = {
    text: string;
    url: string;
};

export type Fix = {
    title: string;
    edits: MessageWithLocation[];
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
    fixes?: Fix[];
};

export type ResultLineSource = {
    file: string | null;
    line: number;
    mainsource?: boolean;
};

export type ResultLine = {
    text: string;
    tag?: ResultLineTag;
    source?: ResultLineSource;
    line?: number; // todo: this should not exist
};
