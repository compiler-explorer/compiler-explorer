
export class PascalUtils {
    isProgram(source) {
        const re = /^\s*program\s+([\w\d_-]*);/ig;
        return !!source.match(re);
    }

    isUnit(source) {
        const re = /^\s*unit\s+([\w\d_-]*);/ig;
        return !!source.match(re);
    }
}
