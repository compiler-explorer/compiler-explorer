// Taken from https://github.com/Nanonid/rison at e64af6c096fd30950ec32cfd48526ca6ee21649d (Jun 9, 2017)
// Uses CommonJS, AMD or browser globals to create a module.
// Based on: https://github.com/umdjs/umd/blob/master/commonjsStrict.js

import * as Sentry from '@sentry/browser';

import {unwrap} from './assert';

//////////////////////////////////////////////////
//
//  the stringifier is based on
//    http://json.org/json.js as of 2006-04-28 from json.org
//  the parser is based on
//    http://osteele.com/sources/openlaszlo/json
//

/**
 *  rules for an uri encoder that is more tolerant than encodeURIComponent
 *
 *  encodeURIComponent passes  ~!*()-_.'
 *
 *  we also allow              ,:@$/
 *
 */
const uri_ok = {
    // ok in url paths and in form query args
    '~': true,
    '!': true,
    '*': true,
    '(': true,
    ')': true,
    '-': true,
    _: true,
    '.': true,
    ',': true,
    ':': true,
    '@': true,
    $: true,
    "'": true,
    '/': true,
};

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

const _idrx = '[^' + not_idstart + not_idchar + '][^' + not_idchar + ']*';

const id_ok = new RegExp('^' + _idrx + '$');

// regexp to find the end of an id when parsing
// g flag on the regexp is necessary for iterative regexp.exec()
const next_id = new RegExp(_idrx, 'g');

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
const sq = {
    "'": true,
    '!': true,
};
const s = {
    array: function (x) {
        const a = ['!('];
        let b;
        let f;
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
    },
    boolean: function (x) {
        if (x) return '!t';
        return '!f';
    },
    null: function () {
        return '!n';
    },
    number: function (x) {
        if (!isFinite(x)) return '!n';
        // strip '+' out of exponent, '-' is ok though
        return String(x).replace(/\+/, '');
    },
    object: function (x) {
        if (x) {
            if (x instanceof Array) {
                return s.array(x);
            }
            // WILL: will this work on non-Firefox browsers?
            if (typeof x.__prototype__ === 'object' && typeof x.__prototype__.encode_rison !== 'undefined')
                return x.encode_rison();

            const a = ['('];
            let b;
            let i;
            let v;
            let k;
            let ki;
            const ks: any[] = [];
            for (i in x) ks[ks.length] = i;
            ks.sort();
            for (ki = 0; ki < ks.length; ki++) {
                i = ks[ki];
                v = enc(x[i]);
                if (typeof v == 'string') {
                    if (b) {
                        a[a.length] = ',';
                    }
                    k = isNaN(parseInt(i)) ? s.string(i) : s.number(i);
                    a.push(k, ':', v);
                    b = true;
                }
            }
            a[a.length] = ')';
            return a.join('');
        }
        return '!n';
    },
    string: function (x) {
        if (x === '') return "''";

        if (id_ok.test(x)) return x;

        x = x.replace(/(['!])/g, function (a, b) {
            if (sq[b]) return '!' + b;
            return b;
        });
        return "'" + x + "'";
    },
    undefined: function () {
        // ignore undefined just like JSON
    },
};

function enc(v: any) {
    if (v && typeof v.toJSON === 'function') v = v.toJSON();
    const fn = s[typeof v];
    if (fn) return fn(v);
}

/**
 * rison-encode a javascript structure
 *
 *  implemementation based on Douglas Crockford's json.js:
 *    http://json.org/json.js as of 2006-04-28 from json.org
 *
 */
export function encode(v: any) {
    return enc(v);
}

/**
 * rison-encode a javascript object without surrounding parens
 *
 */
export function encode_object(v: any) {
    if (typeof v != 'object' || v === null || v instanceof Array)
        throw new Error('rison.encode_object expects an object argument');
    const r = s[typeof v](v);
    return r.substring(1, r.length - 1);
}

/**
 * rison-encode a javascript array without surrounding parens
 *
 */
export function encode_array(v: any) {
    if (!(v instanceof Array)) throw new Error('rison.encode_array expects an array argument');
    const r = s[typeof v](v);
    return r.substring(2, r.length - 1);
}

/**
 * rison-encode and uri-encode a javascript structure
 *
 */
export function encode_uri(v: any) {
    return quote(s[typeof v](v));
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
    const errcb = function (e) {
        throw Error('rison decoder error: ' + e);
    };
    const p = new parser(errcb);
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

class parser {
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
        '(': parser.parse_array,
    };

    string: string;
    index: number;
    message: string | null;
    readonly table: Record<string, () => void>;

    constructor(private errorHandler: (err: string, index: number) => void) {
        this.string = '';
        this.index = -1;
        this.message = null;
        this.table = {
            '!': () => {
                const s = this.string;
                const c = s.charAt(this.index++);
                if (!c) return this.error('"!" at end of input');
                const x = parser.bangs[c];
                if (typeof x == 'function') {
                    // eslint-disable-next-line no-useless-call
                    return x.call(null, this);
                } else if (typeof x == 'undefined') {
                    return this.error('unknown literal: "!' + c + '"');
                }
                return x;
            },
            '(': () => {
                const o = {};
                let c;
                let count = 0;
                while ((c = this.next()) !== ')') {
                    if (count) {
                        if (c !== ',') this.error("missing ','");
                    } else if (c === ',') {
                        return this.error("extra ','");
                    } else --this.index;
                    const k = this.readValue();
                    if (typeof k == 'undefined') return undefined;
                    if (this.next() !== ':') return this.error("missing ':'");
                    const v = this.readValue();
                    if (typeof v == 'undefined') return undefined;
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
                    if (!c) return this.error('unmatched "\'"');
                    if (c === '!') {
                        if (start < i - 1) segments.push(s.slice(start, i - 1));
                        c = s.charAt(i++);
                        if ("!'".indexOf(c) >= 0) {
                            segments.push(c);
                        } else {
                            return this.error('invalid string escape: "!' + c + '"');
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
                    if (permittedSigns.indexOf(c) >= 0) {
                        permittedSigns = '';
                        continue;
                    }
                    state = transitions[state + '+' + c.toLowerCase()];
                    if (state === 'exp') permittedSigns = '-';
                } while (state);
                this.index = --i;
                s = s.slice(start, i);
                if (s === '-') return this.error('invalid number');
                return Number(s);
            },
        };
        // copy table['-'] to each of table[i] | i <- '0'..'9':
        for (let i = 0; i <= 9; i++) this.table[String(i)] = this.table['-'];
    }

    // expose this as-is?
    setOptions(options: {errorHandler?: (err: string, index: number) => void}) {
        if (options['errorHandler']) this.errorHandler = options.errorHandler;
    }

    /**
     * parse a rison string into a javascript structure.
     */
    parse(str) {
        this.string = str;
        this.index = 0;
        this.message = null;
        let value = this.readValue();
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (!this.message && this.next()) value = this.error("unable to parse string as rison: '" + encode(str) + "'");
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (this.message && this.errorHandler) this.errorHandler(this.message, this.index);
        return value;
    }

    error(message: string) {
        Sentry.captureMessage('rison parser error: ' + message);
        this.message = message;
        return undefined;
    }

    readValue() {
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

        if (c) return this.error("invalid character: '" + c + "'");
        return this.error('empty expression');
    }

    // return the next non-whitespace character, or undefined
    next() {
        let c;
        const s = this.string;
        let i = this.index;
        do {
            if (i === s.length) return undefined;
            c = s.charAt(i++);
        } while (parser.WHITESPACE.indexOf(c) >= 0);
        this.index = i;
        return c;
    }

    static parse_array(parser) {
        const ar: any[] = [];
        let c;
        while ((c = parser.next()) !== ')') {
            if (!c) return parser.error("unmatched '!('");
            if (ar.length) {
                if (c !== ',') parser.error("missing ','");
            } else if (c === ',') {
                return parser.error("extra ','");
            } else --parser.index;
            const n = parser.readValue();
            if (typeof n == 'undefined') return undefined;
            ar.push(n);
        }
        return ar;
    }
}
