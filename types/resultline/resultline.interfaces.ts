export interface IResultLineTag {
    line?: number;
    column?: number;
    file?: string;
    text: string;
}

export interface IResultLine {
    text: string;
    tag?: IResultLineTag;
}
