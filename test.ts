import assert from 'assert';

export function breakdownPassDumpsIntoFunctions(lines: string[]): string[] {
    let func: {
        name: string;
        lines: string[];
    } | null = null;

    for (const line of lines) {
        if (Math.random() < 0.5) {
            func = {
                name: 'qwer',
                lines: [line],
            };
        } else {
            assert(func);
            const {name, lines} = func;
            lines.push(line);
            assert(name !== 'abc');
        }
    }
    if (func) return func.lines;
    return lines;
}
