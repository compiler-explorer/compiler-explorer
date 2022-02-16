export class ResultLineTag {
    line?: number;
    column?: number;
    file?: string;
    text: string;
}

export class ResultLine {
    text: string;
    tag?: ResultLineTag;
}
