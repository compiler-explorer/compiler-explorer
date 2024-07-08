import {assert} from 'console';

export class BytesReader {
    protected buffer: Uint8Array;
    protected pointer: number;
    protected _readed_size = 0;

    bind(array: Uint8Array, seek = 0) {
        this.buffer = array;
        this.pointer = seek;
    }
    seek(n: number) {
        this.pointer = n;
    }
    readed_size() {
        return this._readed_size;
    }
    pos() {
        return this.pointer;
    }
    isEnd() {
        return this.pointer >= this.buffer.length;
    }
    read(n: number) {
        const last = this.pointer + n;
        assert(last <= this.buffer.length);
        const array = this.buffer.slice(this.pointer, last);
        this.pointer = last;
        this._readed_size = n;
        return array;
    }
    readLittleEndian(n: number) {
        const s = this.read(n);
        let result = 0;
        for (let i = 0; i < n; i++) {
            result |= s[i] << (8 * i);
        }
        return result;
    }
    readUnsigned(n: number) {
        return this.readLittleEndian(n);
    }
    readSigned(n: number) {
        const bits = n * 8 - 1;
        const mask = ~(1 << bits);
        const val = this.readLittleEndian(n);
        const abs = val & mask;
        return val >> bits ? abs - (1 << bits) : abs;
    }

    readByte() {
        return this.readUnsigned(1);
    }
    readHalf() {
        return this.readUnsigned(2);
    }
    readWord() {
        return this.readUnsigned(4);
    }

    readSByte() {
        return this.readSigned(1);
    }
    readSHalf() {
        return this.readSigned(2);
    }
    readSWord() {
        return this.readSigned(4);
    }

    readULeb128() {
        let result = 0n;
        let i = 0;
        const current = this.pointer;
        for (; this.buffer[this.pointer + i] >> 7; i++) {
            const big = BigInt(this.buffer[this.pointer + i]);
            result |= (big & 0x7fn) << BigInt(7 * i);
        }
        // last byte
        const big = BigInt(this.buffer[this.pointer + i]);
        result |= (big & 0x7fn) << BigInt(7 * i);
        this.pointer += i + 1;
        this._readed_size = this.pointer - current;
        return result;
    }
    readSLeb128() {
        let result = 0n;
        let i = 0;
        const current = this.pointer;
        for (; this.buffer[this.pointer + i] >> 7; i++) {
            const big = BigInt(this.buffer[this.pointer + i]);
            result |= (big & 0x7fn) << BigInt(7 * i);
        }
        // last byte
        const big = BigInt(this.buffer[this.pointer + i]);
        result |= (big & 0x7fn) << BigInt(7 * i);
        this.pointer += i + 1;
        const bits = i * 7 + 6;
        if (big & 0x40n) {
            result = result - (1n << BigInt(bits + 1));
        }
        this._readed_size = this.pointer - current;
        return result;
    }
    readString() {
        const current = this.pointer;
        let c = this.readByte();
        let result = '';
        while (c !== 0) {
            result += String.fromCodePoint(c);
            c = this.readByte();
        }
        this._readed_size = this.pointer - current;
        return result;
    }
}
