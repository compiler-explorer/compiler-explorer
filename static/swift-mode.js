// With thanks to https://github.com/carabina/vscode-swift/blob/master/swiftDef.js
// (MIT licensed)

define(function (require) {
    'use strict';
    var monaco = require('monaco');

    function definition() {
        return {
            displayName: '',
            name: 'swift',
            mimeTypes: [],
            fileExtensions: [],
            defaultToken: '',
            // used in the editor to insert comments (ctrl+/ or shift+alt+A)
            lineComment: '// ',
            blockCommentStart: '/*',
            blockCommentEnd: '*/',
            // the default separators except `@`
            wordDefinition: /(-?\d*\.\d\w*)|([^`~!#$%\^&*()\-=+\[{\]}\\|;:'",.<>\/?\s]+)/g,
            autoClosingPairs: [
                ['"', '"'],
                ['\'', '\''],
                ['{', '}'],
                ['[', ']'],
                ['(', ')'],
            ],
            brackets: [
                {open: '{', close: '}', token: 'delimiter.curly'},
                {open: '[', close: ']', token: 'delimiter.square'},
                {open: '(', close: ')', token: 'delimiter.parenthesis'},
                {open: '<', close: '>', token: 'delimiter.angle'}
            ],
            editorOptions: {tabSize: 4, insertSpaces: true},
            keywords: [
                "__COLUMN__",
                "__FILE__",
                "__FUNCTION__",
                "__LINE__",
                "as",
                "associativity",
                "break",
                "case",
                "class",
                "continue",
                "convenience",
                "default",
                "deinit",
                "didSet",
                "do",
                "dynamic",
                "dynamicType",
                "else",
                "enum",
                "extension",
                "fallthrough",
                "final",
                "for",
                "func",
                "get",
                "if",
                "import",
                "in",
                "infix",
                "init",
                "inout",
                "internal",
                "is",
                "lazy",
                "left",
                "let",
                "mutating",
                "nil",
                "none",
                "nonmutating",
                "operator",
                "optional",
                "override",
                "postfix",
                "precedence",
                "prefix",
                "private",
                "protocol",
                "Protocol",
                "public",
                "required",
                "return",
                "right",
                "self",
                "Self",
                "set",
                "static",
                "struct",
                "subscript",
                "super",
                "switch",
                "Type",
                "typealias",
                "unowned",
                "var",
                "weak",
                "where",
                "while",
                "willSet",
                "FALSE",
                "TRUE",
            ],
//        namespaceFollows: [
//            'namespace',
//            'using',
//        ],
//        parenFollows: [
//            'if',
//            'for',
//            'while',
//            'switch',
//            'foreach',
//            'using',
//            'catch'
//        ],
            operators: [
                '=',
                '??',
                '||',
                '&&',
                '|',
                '^',
                '&',
                '==',
                '!=',
                '<=',
                '>=',
                '<<',
                '+',
                '-',
                '*',
                '/',
                '%',
                '!',
                '~',
                '++',
                '--',
                '+=',
                '-=',
                '*=',
                '/=',
                '%=',
                '&=',
                '|=',
                '^=',
                '<<=',
                '>>=',
                '>>',
                '=>'
            ],
            symbols: /[=><!~?:&|+\-*\/\^%]+/,
            // escape sequences
            escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,
            // The main tokenizer for our languages
            tokenizer: {
                root: [
                    [/\@?[a-zA-Z_]\w*/, {
                        cases: {
//                    '@namespaceFollows': { token: 'keyword.$0', next: '@namespace' },
                            '@keywords': {token: 'keyword.$0', next: '@qualified'},
                            '@default': {token: 'identifier', next: '@qualified'}
                        }
                    }],
                    {include: '@whitespace'},
                    [/}/, {
                        cases: {
                            '$S2==interpolatedstring': {token: 'string.quote', bracket: '@close', next: '@pop'},
                            '@default': '@brackets'
                        }
                    }],
                    [/[{}()\[\]]/, '@brackets'],
                    [/[<>](?!@symbols)/, '@brackets'],
                    [/@symbols/, {cases: {'@operators': 'delimiter', '@default': ''}}],
                    [/\@"/, {token: 'string.quote', bracket: '@open', next: '@litstring'}],
                    [/\$"/, {token: 'string.quote', bracket: '@open', next: '@interpolatedstring'}],
                    [/\d*\.\d+([eE][\-+]?\d+)?[fFdD]?/, 'number.float'],
                    [/0[xX][0-9a-fA-F]+/, 'number.hex'],
                    [/\d+/, 'number'],
                    [/[;,.]/, 'delimiter'],
                    [/"([^"\\]|\\.)*$/, 'string.invalid'],
                    [/"/, {token: 'string.quote', bracket: '@open', next: '@string'}],
                    [/'[^\\']'/, 'string'],
                    [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                    [/'/, 'string.invalid']
                ],
                qualified: [
                    [/[a-zA-Z_][\w]*/, {cases: {'@keywords': {token: 'keyword.$0'}, '@default': 'identifier'}}],
                    [/\./, 'delimiter'],
                    ['', '', '@pop'],
                ],
//            namespace: [
//                { include: '@whitespace' },
//                [/[A-Z]\w*/, 'namespace'],
//                [/[\.=]/, 'delimiter'],
//                ['', '', '@pop'],
//            ],
                comment: [
                    [/[^\/*]+/, 'comment'],
                    ['\\*/', 'comment', '@pop'],
                    [/[\/*]/, 'comment']
                ],
                string: [
                    [/[^\\"]+/, 'string'],
                    [/@escapes/, 'string.escape'],
                    [/\\./, 'string.escape.invalid'],
                    [/"/, {token: 'string.quote', bracket: '@close', next: '@pop'}]
                ],
                litstring: [
                    [/[^"]+/, 'string'],
                    [/""/, 'string.escape'],
                    [/"/, {token: 'string.quote', bracket: '@close', next: '@pop'}]
                ],
                interpolatedstring: [
                    [/[^\\"{]+/, 'string'],
                    [/@escapes/, 'string.escape'],
                    [/\\./, 'string.escape.invalid'],
                    [/{{/, 'string.escape'],
                    [/}}/, 'string.escape'],
                    [/{/, {token: 'string.quote', bracket: '@open', next: 'root.interpolatedstring'}],
                    [/"/, {token: 'string.quote', bracket: '@close', next: '@pop'}]
                ],
                whitespace: [
                    [/^[ \t\v\f]*#\w.*$/, 'namespace.cpp'],
                    [/[ \t\v\f\r\n]+/, ''],
                    [/\/\*/, 'comment', '@comment'],
                    [/\/\/.*$/, 'comment'],
                ],
            },
        };
    }

    monaco.languages.register({id: 'swift'});
    monaco.languages.setMonarchTokensProvider('swift', definition());
});