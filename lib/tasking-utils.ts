import {ResultLine, ResultLineTag} from '../types/resultline/resultline.interfaces';

const errRe = /^([aclprs]{0,2}tc)\s+(\w+):\s*\["(.+)"\s+(\d+)\/?(\d+)?]\s*(.+)\s*$/;

export function parseError(input: string, inputFilename?: string, pathPrefix?: string): ResultLine[] {
    const lines = input.split('\n');
    const result: ResultLine[] = [];
    for (const line of lines) {
        const ma_res = line.match(errRe);
        if (!ma_res) {
            result.push({text: line});
            continue;
        }
        const compiler = ma_res[1];
        const errno = ma_res[2];
        const src = ma_res[3] === inputFilename ? '<source>' : ma_res[3];
        const lineno = parseInt(ma_res[4]);
        const column = ma_res[5] ? parseInt(ma_res[5]) : NaN;
        const message = ma_res[6];
        const tag: ResultLineTag = {
            text: message,
            severity: errno.startsWith('E') ? 3 : 1,
            line: lineno,
            file: src.substring(src.lastIndexOf('\\') + 1),
        };
        if (!Number.isNaN(column)) {
            tag.column = column;
        }
        result.push({
            text: `${compiler} ${errno}: [${src} ${lineno}/${column}] ${message}`,
            tag: tag,
        });
    }
    return result;
}
