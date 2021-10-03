/* eslint-disable header/header */

// Copyright (c) 2012 Rob Burns
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

// Converted from https://github.com/rburns/ansi-to-html
// Includes patches from https://github.com/rburns/ansi-to-html/pull/84
// Converted to typescript by MarkusJx

'use strict';

import _ from 'underscore';
import {AnsiToHtmlOptions, ColorCodes} from './ansi-to-html.interfaces';

const defaults: AnsiToHtmlOptions = {
    fg: '#FFF',
    bg: '#000',
    newline: false,
    escapeXML: false,
    stream: false,
    colors: getDefaultColors(),
};

function getDefaultColors(): ColorCodes {
    const colors: ColorCodes = {
        0: '#000',
        1: '#A00',
        2: '#0A0',
        3: '#A50',
        4: '#00A',
        5: '#A0A',
        6: '#0AA',
        7: '#AAA',
        8: '#555',
        9: '#F55',
        10: '#5F5',
        11: '#FF5',
        12: '#55F',
        13: '#F5F',
        14: '#5FF',
        15: '#FFF',
    };

    range(0, 5).forEach(function(red: number) {
        range(0, 5).forEach(function(green: number) {
            range(0, 5).forEach(function(blue: number) {
                return setStyleColor(red, green, blue, colors);
            });
        });
    });

    range(0, 23).forEach(function(gray: number) {
        const c: number = gray + 232;
        const l: string = toHexString(gray * 10 + 8);

        colors[c] = '#' + l + l + l;
    });

    return colors;
}

function setStyleColor(red: number, green: number, blue: number, colors: ColorCodes): void {
    const c: number = 16 + red * 36 + green * 6 + blue;
    const r: number = red > 0 ? red * 40 + 55 : 0;
    const g: number = green > 0 ? green * 40 + 55 : 0;
    const b: number = blue > 0 ? blue * 40 + 55 : 0;

    colors[c] = toColorHexString([r, g, b]);
}

/**
 * Converts from a number like 15 to a hex string like 'F'
 *
 * @param num the number to convert
 * @returns the resulting hex string
 */
function toHexString(num: number): string {
    let str: string = num.toString(16);

    while (str.length < 2) {
        str = '0' + str;
    }

    return str;
}

/**
 * Converts from an array of numbers like [15, 15, 15] to a hex string like 'FFF'
 *
 * @param ref the array of numbers to join
 * @returns the resulting hex string
 */
function toColorHexString(ref: number[]): string {
    const results: string[] = [];

    for (let j = 0, len = ref.length; j < len; j++) {
        results.push(toHexString(ref[j]));
    }

    return '#' + results.join('');
}

function generateOutput(stack: string[], token: string, data: string | number, options: AnsiToHtmlOptions): string {
    let result: string;

    if (token === 'text') {
        // Note: Param 'data' must be a string at this point
        result = pushText(data as string, options);
    } else if (token === 'display') {
        result = handleDisplay(stack, data, options);
    } else if (token === 'xterm256') {
        // Note: Param 'data' must be a string at this point
        result = handleXterm256(stack, data as string, options);
    }

    return result;
}

function handleXterm256(stack: string[], data: string, options: AnsiToHtmlOptions): string {
    data = data.substring(2).slice(0, -1);
    const operation: number = +data.substr(0, 2);
    const color: number = +data.substr(5);
    if (operation === 38) {
        return pushForegroundColor(stack, options.colors[color]);
    } else {
        return pushBackgroundColor(stack, options.colors[color]);
    }
}

function handleDisplay(stack: string[], _code: string | number, options: AnsiToHtmlOptions): string {
    const code: number = parseInt(_code as string, 10);
    let result: string;

    const codeMap: Record<number, () => string> = {
        '-1': function _() {
            return '<br/>';
        },
        0: function _() {
            return stack.length && resetStyles(stack);
        },
        1: function _() {
            return pushTag(stack, 'b');
        },
        2: function _() {
            return pushStyle(stack, 'opacity:0.6');
        },
        3: function _() {
            return pushTag(stack, 'i');
        },
        4: function _() {
            return pushTag(stack, 'u');
        },
        8: function _() {
            return pushStyle(stack, 'display:none');
        },
        9: function _() {
            return pushTag(stack, 'strike');
        },
        22: function _() {
            return closeTag(stack, 'b');
        },
        23: function _() {
            return closeTag(stack, 'i');
        },
        24: function _() {
            return closeTag(stack, 'u');
        },
        39: function _() {
            return pushForegroundColor(stack, options.fg);
        },
        49: function _() {
            return pushBackgroundColor(stack, options.bg);
        },
    };

    if (codeMap[code]) {
        result = codeMap[code]();
    } else if (4 < code && code < 7) {
        result = pushTag(stack, 'blink');
    } else if (29 < code && code < 38) {
        result = pushForegroundColor(stack, options.colors[code - 30]);
    } else if (39 < code && code < 48) {
        result = pushBackgroundColor(stack, options.colors[code - 40]);
    } else if (89 < code && code < 98) {
        result = pushForegroundColor(stack, options.colors[8 + (code - 90)]);
    } else if (99 < code && code < 108) {
        result = pushBackgroundColor(stack, options.colors[8 + (code - 100)]);
    }

    return result;
}

/**
 * Clear all the styles
 */
function resetStyles(stack: string[]): string {
    const stackClone: string[] = stack.slice(0);

    stack.length = 0;

    return stackClone.reverse().map(function(tag: string) {
        return '</' + tag + '>';
    }).join('');
}

/**
 * Creates an array of numbers ranging from low to high
 *
 * @param low the lowest number in the array to create
 * @param high the highest number in the array to create
 * @returns the resulting array
 * @example range(3, 7); // creates [3, 4, 5, 6, 7]
 */
function range(low: number, high: number): number[] {
    const results: number[] = [];

    for (let j: number = low; j <= high; j++) {
        results.push(j);
    }

    return results;
}

/**
 * Returns a new function that is true if value is NOT the same category
 */
function notCategory(category: string): (e: StickyStackElement) => boolean {
    return function(e: StickyStackElement): boolean {
        return (category === null || e.category !== category) && category !== 'all';
    };
}

/**
 * Converts a code into an ansi token type
 *
 * @param _code the code to convert
 * @returns the ansi token type
 */
function categoryForCode(_code: string | number): string {
    const code: number = parseInt(_code as string, 10);
    let result: string = null;

    if (code === 0) {
        result = 'all';
    } else if (code === 1) {
        result = 'bold';
    } else if (2 < code && code < 5) {
        result = 'underline';
    } else if (4 < code && code < 7) {
        result = 'blink';
    } else if (code === 8) {
        result = 'hide';
    } else if (code === 9) {
        result = 'strike';
    } else if (29 < code && code < 38 || code === 39 || 89 < code && code < 98) {
        result = 'foreground-color';
    } else if (39 < code && code < 48 || code === 49 || 99 < code && code < 108) {
        result = 'background-color';
    }

    return result;
}

function pushText(text: string, options: AnsiToHtmlOptions): string {
    if (options.escapeXML) {
        return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    return text;
}

function pushTag(stack: string[], tag: string, style?: string): string {
    if (!style) {
        style = '';
    }

    stack.push(tag);

    return ['<' + tag, style ? ' style="' + style + '"' : void 0, '>'].join('');
}

function pushStyle(stack: string[], style: string): string {
    return pushTag(stack, 'span', style);
}

function pushForegroundColor(stack: string[], color: string): string {
    return pushTag(stack, 'span', 'color:' + color);
}

function pushBackgroundColor(stack: string[], color: string): string {
    return pushTag(stack, 'span', 'background-color:' + color);
}

function closeTag(stack: string[], style: string): string {
    let last: string = null;

    if (stack.slice(-1)[0] === style) {
        last = stack.pop();
    }

    if (last) {
        return '</' + style + '>';
    }
}

type TokenizeCallback = (token: string, data: string | number) => void;

function tokenize(text: string, options: AnsiToHtmlOptions, callback: TokenizeCallback) {
    var ansiMatch: boolean = false;
    var ansiHandler: number = 3;

    function remove(): string {
        return '';
    }

    function removeXterm256(m: string): string {
        callback('xterm256', m);
        return '';
    }

    function newline(m: string): string {
        if (options.newline) {
            callback('display', -1);
        } else {
            callback('text', m);
        }

        return '';
    }

    function ansiMess(m: string, g1: string): string {
        ansiMatch = true;
        if (g1.trim().length === 0) {
            g1 = '0';
        }

        let res: string[] = g1.replace(/;+$/, '').split(';');

        for (let o: number = 0, len: number = res.length; o < len; o++) {
            callback('display', res[o]);
        }

        return '';
    }

    function realText(m: string): string {
        callback('text', m);

        return '';
    }

    type Token = {
        pattern: RegExp,
        sub: (m: string, ...args: any[]) => string
    };

    /* eslint no-control-regex:0 */
    var tokens: Token[] = [{
        pattern: /^\x08+/,
        sub: remove,
    }, {
        pattern: /^\x1b\[[012]?K/,
        sub: remove,
    }, {
        pattern: /^\x1b\[[34]8;5;(\d+)m/,
        sub: removeXterm256,
    }, {
        pattern: /^\n/,
        sub: newline,
    }, {
        pattern: /^\x1b\[((?:\d{1,3};)*\d{1,3}|)m/,
        sub: ansiMess,
    }, {
        pattern: /^\x1b\[?[\d;]{0,3}/,
        sub: remove,
    }, {
        pattern: /^([^\x1b\x08\n]+)/,
        sub: realText,
    }];

    function process(handler: Token, i: number): void {
        if (i > ansiHandler && ansiMatch) {
            return;
        }

        ansiMatch = false;

        text = text.replace(handler.pattern, handler.sub);
    }

    var handler: Token;
    var results1: number[] = [];
    var length: number = text.length;

    outer: while (length > 0) {
        for (let i: number = 0, o: number = 0, len: number = tokens.length; o < len; i = ++o) {
            handler = tokens[i];
            process(handler, i);

            if (text.length !== length) {
                // We matched a token and removed it from the text. We need to
                // start matching *all* tokens against the new text.
                length = text.length;
                continue outer;
            }
        }

        if (text.length === length) {
            break;
        } else {
            results1.push(0);
        }

        length = text.length;
    }

    return results1;
}

/**
 * A sticky stack element
 */
type StickyStackElement = {
    token: string,
    data: number | string,
    category: string
};

/**
 * If streaming, then the stack is "sticky"
 */
function updateStickyStack(stickyStack: StickyStackElement[], token: string, data: string | number): StickyStackElement[] {
    if (token !== 'text') {
        stickyStack = stickyStack.filter(notCategory(categoryForCode(data)));
        stickyStack.push({
            token: token,
            data: data,
            category: categoryForCode(data),
        });
    }

    return stickyStack;
}

export default class Filter {
    private readonly opts: AnsiToHtmlOptions;
    private readonly stack: string[];
    private stickyStack: StickyStackElement[];

    public constructor(options: AnsiToHtmlOptions) {
        options = options || {};

        if (options.colors) {
            options.colors = _.extend(defaults.colors, options.colors);
        }

        this.opts = _.extend({}, defaults, options);
        this.stack = [];
        this.stickyStack = [];
    }

    public toHtml(_input: string | string[]): string {
        const _this = this;

        const input: string[] = typeof _input === 'string' ? [_input] : _input;
        const stack: string[] = this.stack;
        const options: AnsiToHtmlOptions = this.opts;
        const buf: string[] = [];

        this.stickyStack.forEach(function(element: StickyStackElement) {
            const output: string = generateOutput(stack, element.token, element.data, options);

            if (output) {
                buf.push(output);
            }
        });

        tokenize(input.join(''), options, function(token: string, data: string | number) {
            let output: string = generateOutput(stack, token, data, options);

            if (output) {
                buf.push(output);
            }

            if (options.stream) {
                _this.stickyStack = updateStickyStack(_this.stickyStack, token, data);
            }
        });

        if (stack.length) {
            buf.push(resetStyles(stack));
        }

        return buf.join('');
    }
}
