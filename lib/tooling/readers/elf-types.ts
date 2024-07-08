export type uByte = number;
export type sByte = number;
export type uHalf = number;
export type uWord = number;
export type sWord = number;
export type Addr = bigint;
export type uLeb128 = bigint;
export type ByteArray = Uint8Array;

export function signedLeb128(value: uLeb128) {
    value = BigInt(value);
    const bits = value.toString(2).length;
    if (bits % 7 !== 0) {
        return value;
    }
    return value - (1n << BigInt(bits));
}
