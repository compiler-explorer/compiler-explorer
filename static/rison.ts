// Based on https://github.com/Nanonid/rison at e64af6c096fd30950ec32cfd48526ca6ee21649d (Jun 9, 2017)

import {assert, unwrap} from './assert.js';

import {isString} from '../shared/common-utils.js';

//////////////////////////////////////////////////
//
//  the stringifier is based on
//    http://json.org/json.js as of 2006-04-28 from json.org
//  the parser is based on
//    http://osteele.com/sources/openlaszlo/json
//

/*
 * we divide the uri-safe glyphs into three sets
 *   <rison> - used by rison                         ' ! : ( ) ,
 *   <reserved> - not common in strings, reserved    * @ $ & ; =
 *
 * we define <identifier> as anything that's not forbidden
 */

/**
 * punctuation characters that are legal inside ids.
 */
// this var isn't actually used
//rison.idchar_punctuation = "_-./~";

const not_idchar = " '!:(),*@$";

/**
 * characters that are illegal as the start of an id
 * this is so ids can't look like numbers.
 */
const not_idstart = '-0123456789';

const [id_ok, next_id] = (() => {
    const _idrx = '[^' + not_idstart + not_idchar + '][^' + not_idchar + ']*';
    return [
        new RegExp('^' + _idrx + '$'),
        // regexp to find the end of an id when parsing
        // g flag on the regexp is necessary for iterative regexp.exec()
        new RegExp(_idrx, 'g'),
    ];
})();

/**
 * this is like encodeURIComponent() but quotes fewer characters.
 *
 * @see rison.uri_ok
 *
 * encodeURIComponent passes   ~!*()-_.'
 * rison.quote also passes   ,:@$/
 *   and quotes " " as "+" instead of "%20"
 */
export function quote(x: string) {
    if (/^[-A-Za-z0-9~!*()_.',:@$/]*$/.test(x)) return x;

    return encodeURIComponent(x)
        .replace(/%2C/g, ',')
        .replace(/%3A/g, ':')
        .replace(/%40/g, '@')
        .replace(/%24/g, '$')
        .replace(/%2F/g, '/')
        .replace(/%20/g, '+');
}

//
//  based on json.js 2006-04-28 from json.org
//  license: http://www.json.org/license.html
//
//  hacked by nix for use in uris.
//
// url-ok but quoted in strings
const string_table = {
    "'": true,
    '!': true,
};

class Encoders {
    static array(x: JSONValue[]) {
        const a = ['!('];
        let b;
        let i;
        const l = x.length;
        let v;
        for (i = 0; i < l; i += 1) {
            v = enc(x[i]);
            if (typeof v == 'string') {
                if (b) {
                    a[a.length] = ',';
                }
                a[a.length] = v;
                b = true;
            }
        }
        a[a.length] = ')';
        return a.join('');
    }
    static boolean(x: boolean) {
        if (x) return '!t';
        return '!f';
    }
    static null() {
        return '!n';
    }
    static number(x: number) {
        if (!isFinite(x)) return '!n';
        // strip '+' out of exponent, '-' is ok though
        return String(x).replace(/\+/, '');
    }
    static object(x: Record<string, JSONValue> | null) {
        if (x) {
            // because typeof null === 'object'
            if (x instanceof Array) {
                return Encoders.array(x);
            }

            const a = ['('];
            let b = false;
            let i: string;
            let v: string | undefined;
            let k: string;
            let ki: number;
            const ks: string[] = [];
            for (const i in x) ks[ks.length] = i;
            ks.sort();
            for (ki = 0; ki < ks.length; ki++) {
                i = ks[ki];
                v = enc(x[i]);
                if (typeof v == 'string') {
                    if (b) {
                        a[a.length] = ',';
                    }
                    k = isNaN(parseInt(i)) ? Encoders.string(i) : Encoders.number(parseInt(i));
                    a.push(k, ':', v);
                    b = true;
                }
            }
            a[a.length] = ')';
            return a.join('');
        }
        return '!n';
    }
    static string(x: string) {
        if (x === '') return "''";

        if (id_ok.test(x)) return x;

        x = x.replace(/(['!])/g, function (a, b) {
            if (string_table[b]) return '!' + b;
            return b;
        });
        return "'" + x + "'";
    }
    static undefined() {
        // ignore undefined just like JSON
        return undefined;
    }
}

const encode_table: Record<string, (x: any) => string | undefined> = {
    array: Encoders.array,
    object: Encoders.object,
    boolean: Encoders.boolean,
    string: Encoders.string,
    number: Encoders.number,
    null: Encoders.null,
    undefined: Encoders.undefined,
};

function enc(v: JSONValue | (JSONValue & {toJSON?: () => string})) {
    if (v && typeof v === 'object' && 'toJSON' in v && typeof v.toJSON === 'function') v = v.toJSON();
    if ((typeof v) in encode_table) {
        return encode_table[typeof v](v);
    }
}

/**
 * rison-encode a javascript structure
 *
 *  implemementation based on Douglas Crockford's json.js:
 *    http://json.org/json.js as of 2006-04-28 from json.org
 *
 */
export function encode(v: JSONValue | (JSONValue & {toJSON?: () => string})) {
    return enc(v);
}

/**
 * rison-encode a javascript object without surrounding parens
 *
 */
export function encode_object(v: JSONValue) {
    if (typeof v != 'object' || v === null || v instanceof Array)
        throw new Error('rison.encode_object expects an object argument');
    const r = unwrap(encode_table[typeof v](v));
    return r.substring(1, r.length - 1);
}

/**
 * rison-encode a javascript array without surrounding parens
 *
 */
export function encode_array(v: JSONValue) {
    if (!(v instanceof Array)) throw new Error('rison.encode_array expects an array argument');
    const r = unwrap(encode_table[typeof v](v));
    return r.substring(2, r.length - 1);
}

/**
 * rison-encode and uri-encode a javascript structure
 *
 */
export function encode_uri(v: JSONValue) {
    return quote(unwrap(encode_table[typeof v](v)));
}

//
// based on openlaszlo-json and hacked by nix for use in uris.
//
// Author: Oliver Steele
// Copyright: Copyright 2006 Oliver Steele.  All rights reserved.
// Homepage: http://osteele.com/sources/openlaszlo/json
// License: MIT License.
// Version: 1.0

/**
 * parse a rison string into a javascript structure.
 *
 * this is the simplest decoder entry point.
 *
 *  based on Oliver Steele's OpenLaszlo-JSON
 *     http://osteele.com/sources/openlaszlo/json
 */
export function decode(r: string) {
    const p = new Parser();
    return p.parse(r);
}

/**
 * parse an o-rison string into a javascript structure.
 *
 * this simply adds parentheses around the string before parsing.
 */
export function decode_object(r: string) {
    return decode('(' + r + ')');
}

/**
 * parse an a-rison string into a javascript structure.
 *
 * this simply adds array markup around the string before parsing.
 */
export function decode_array(r: string) {
    return decode('!(' + r + ')');
}

// prettier-ignore
export type JSONValue =
    | string
    | number
    | boolean
    | null
    | undefined
    | {[x: string]: JSONValue}
    | Array<JSONValue>;

class Parser {
    /**
     * a string containing acceptable whitespace characters.
     * by default the rison decoder tolerates no whitespace.
     * to accept whitespace set rison.parser.WHITESPACE = " \t\n\r\f";
     */
    static WHITESPACE = '';

    static readonly bangs = {
        t: true,
        f: false,
        n: null,
        '(': Parser.parse_array,
    };

    string: string;
    index: number;
    readonly table: Record<string, () => JSONValue>;

    constructor() {
        this.string = '';
        this.index = -1;
        this.table = {
            '!': () => {
                const s = this.string;
                const c = s.charAt(this.index++);
                if (!c) return this.error('"!" at end of input');
                const x = Parser.bangs[c];
                if (typeof x == 'function') {
                    // eslint-disable-next-line no-useless-call
                    return x.call(null, this);
                } else if (typeof x === 'undefined') {
                    return this.error('unknown literal: "!' + c + '"');
                }
                return x;
            },
            '(': () => {
                const o: JSONValue = {};
                let c;
                let count = 0;
                while ((c = this.next()) !== ')') {
                    if (count) {
                        if (c !== ',') this.error("missing ','");
                    } else if (c === ',') {
                        this.error("extra ','");
                    } else --this.index;
                    const k = this.readValue();
                    if (typeof k == 'undefined') return undefined;
                    if (this.next() !== ':') this.error("missing ':'");
                    const v = this.readValue();
                    if (typeof v == 'undefined') return undefined;
                    assert(isString(k));
                    o[k] = v;
                    count++;
                }
                return o;
            },
            "'": () => {
                const s = this.string;
                let i = this.index;
                let start = i;
                const segments: string[] = [];
                let c;
                while ((c = s.charAt(i++)) !== "'") {
                    //if (i == s.length) return this.error('unmatched "\'"');
                    if (!c) this.error('unmatched "\'"');
                    if (c === '!') {
                        if (start < i - 1) segments.push(s.slice(start, i - 1));
                        c = s.charAt(i++);
                        if ("!'".includes(c)) {
                            segments.push(c);
                        } else {
                            this.error('invalid string escape: "!' + c + '"');
                        }
                        start = i;
                    }
                }
                if (start < i - 1) segments.push(s.slice(start, i - 1));
                this.index = i;
                return segments.length === 1 ? segments[0] : segments.join('');
            },
            // Also any digit.  The statement that follows this table
            // definition fills in the digits.
            '-': () => {
                let s = this.string;
                let i = this.index;
                const start = i - 1;
                let state = 'int';
                let permittedSigns = '-';
                const transitions = {
                    'int+.': 'frac',
                    'int+e': 'exp',
                    'frac+e': 'exp',
                };
                do {
                    const c = s.charAt(i++);
                    if (!c) break;
                    if ('0' <= c && c <= '9') continue;
                    if (permittedSigns.includes(c)) {
                        permittedSigns = '';
                        continue;
                    }
                    state = transitions[state + '+' + c.toLowerCase()];
                    if (state === 'exp') permittedSigns = '-';
                } while (state);
                this.index = --i;
                s = s.slice(start, i);
                if (s === '-') this.error('invalid number');
                return Number(s);
            },
        };
        // copy table['-'] to each of table[i] | i <- '0'..'9':
        for (let i = 0; i <= 9; i++) this.table[String(i)] = this.table['-'];
    }

    /**
     * parse a rison string into a javascript structure.
     */
    parse(str: string): JSONValue {
        this.string = str;
        this.index = 0;
        const value = this.readValue();
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (this.next()) this.error("unable to parse string as rison: '" + encode(str) + "'");
        return value;
    }

    error(message: string): never {
        throw new Error('rison parser error: ' + message);
    }

    readValue(): JSONValue {
        const c = this.next();
        const fn = c && this.table[c];

        if (fn) return fn.apply(this);

        // fell through table, parse as an id

        const s = this.string;
        const i = this.index - 1;

        // Regexp.lastIndex may not work right in IE before 5.5?
        // g flag on the regexp is also necessary
        next_id.lastIndex = i;
        const m = unwrap(next_id.exec(s));

        // console.log('matched id', i, r.lastIndex);

        if (m.length > 0) {
            const id = m[0];
            this.index = i + id.length;
            return id; // a string
        }

        if (c) this.error("invalid character: '" + c + "'");
        this.error('empty expression');
    }

    next(): string | undefined {
        let c: string;
        const s = this.string;
        let i = this.index;
        do {
            if (i === s.length) return undefined;
            c = s.charAt(i++);
        } while (Parser.WHITESPACE.includes(c));
        this.index = i;
        return c;
    }

    static parse_array(parser: Parser): JSONValue[] | undefined {
        const ar: JSONValue[] = [];
        let c;
        while ((c = parser.next()) !== ')') {
            if (!c) return parser.error("unmatched '!('");
            if (ar.length) {
                if (c !== ',') parser.error("missing ','");
            } else if (c === ',') {
                return parser.error("extra ','");
            } else --parser.index;
            const n = parser.readValue();
            if (n === undefined) return undefined;
            ar.push(n);
        }
        return ar;
    }
}
