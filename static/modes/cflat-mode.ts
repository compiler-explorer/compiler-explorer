// This file is a customized version of for cflat compiler.
// Take compiler-explorer/static/modes/asm-mode.ts as reference.

// The following interfaces are imported from the shared "monaco-editor" folder.

import * as monaco from 'monaco-editor';

function definition(): monaco.languages.IMonarchLanguage {
    return {
        // Set defaultToken to invalid to see what you do not tokenize yet
        defaultToken: 'invalid',

        // C# style strings
        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        registers: /%?\b(r[0-9]+[dbw]?|([er]?([abcd][xhl]|cs|fs|ds|ss|sp|bp|ip|sil?|dil?))|[xyz]mm[0-9]+|sp|fp|lr)\b/,

        intelOperators: /PTR|(D|Q|[XYZ]MM)?WORD/,

        tokenizer: {
            root: [
                // Error document
                [/^<.*>$/, {token: 'annotation'}],
                // inline comments
                [/\/\*/, 'comment', '@comment'],
                // Label definition
                [/^[.a-zA-Z0-9_$?@].*:/, {token: 'type.identifier'}],
                // Label definition (quoted)
                [/^"([^"\\]|\\.)*":/, {token: 'type.identifier'}],
                // Label definition (ARM style)
                [/^\s*[|][^|]*[|]/, {token: 'type.identifier'}],
                // Label definition (CL style)
                [/^\s*[.a-zA-Z0-9_$|]*\s+(PROC|ENDP|DB|DD)/, {token: 'type.identifier', next: '@rest'}],
                // Constant definition
                [/^[.a-zA-Z0-9_$?@][^=]*=/, {token: 'type.identifier'}],
                // opcode
                [/[.a-zA-Z_][.a-zA-Z_0-9]*/, {token: 'keyword', next: '@rest'}],
                // braces and parentheses at the start of the line (e.g. nvcc output)
                [/[(){}]/, {token: 'operator', next: '@rest'}],
                // msvc can have strings at the start of a line in a inSegDirList
                [/`/, {token: 'string.backtick', bracket: '@open', next: '@segDirMsvcstring'}],

                // whitespace
                {include: '@whitespace'},
            ],

            rest: [
                // pop at the beginning of the next line and rematch
                [/^.*$/, {token: '@rematch', next: '@pop'}],

                [/@registers/, 'variable.predefined'],
                [/@intelOperators/, 'annotation'],
                // inline comments
                [/\/\*/, 'comment', '@comment'],

                // brackets
                [/[{}<>()[\]]/, '@brackets'],

                // ARM-style label reference
                [/[|][^|]*[|]*/, 'type.identifier'],

                // numbers
                [/\d*\.\d+([eE][-+]?\d+)?/, 'number.float'],
                [/([$]|0[xX])[0-9a-fA-F]+/, 'number.hex'],
                [/\d+/, 'number'],
                // ARM-style immediate numbers (which otherwise look like comments)
                [/#-?\d+/, 'number'],

                // operators
                [/[-+,*/!:&{}()]/, 'operator'],

                // strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-terminated string
                [/"/, {token: 'string.quote', bracket: '@open', next: '@string'}],
                // `msvc does this, sometimes'
                [/`/, {token: 'string.backtick', bracket: '@open', next: '@msvcstring'}],
                [/'/, {token: 'string.singlequote', bracket: '@open', next: '@sstring'}],

                // characters
                [/'[^\\']'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],

                // Assume anything else is a label reference. .NET uses ` in some identifiers
                [/%?[.?_$a-zA-Z@][.?_$a-zA-Z0-9@`]*/, 'type.identifier'],

                // whitespace
                {include: '@whitespace'},
            ],

            comment: [
                [/[^/*]+/, 'comment'],
                [/\/\*/, 'comment', '@push'], // nested comment
                ['\\*/', 'comment', '@pop'],
                [/[/*]/, 'comment'],
            ],

            string: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, {token: 'string.quote', bracket: '@close', next: '@pop'}],
            ],

            msvcstringCommon: [
                [/[^\\']+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/''/, 'string.escape'], // ` isn't escaped but ' is escaped as ''
                [/\\./, 'string.escape.invalid'],
            ],

            msvcstring: [
                {include: '@msvcstringCommon'},
                [/'/, {token: 'string.backtick', bracket: '@close', next: '@pop'}],
            ],

            segDirMsvcstring: [
                {include: '@msvcstringCommon'},
                [/'/, {token: 'string.backtick', bracket: '@close', switchTo: '@rest'}],
            ],

            sstring: [
                [/[^\\']+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/'/, {token: 'string.singlequote', bracket: '@close', next: '@pop'}],
            ],

            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/\/\*/, 'comment', '@comment'],
                [/\/\/.*$/, 'comment'],
                [/[#;\\@].*$/, 'comment'],
            ],
        },
    };
}

const def = definition();
monaco.languages.register({id: 'cflat'});
monaco.languages.setMonarchTokensProvider('cflat', def);

export = def;
