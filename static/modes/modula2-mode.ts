// Copyright (c) 2018, Eugen Bulavin
// Copyright (c) 2023, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
import * as monaco from 'monaco-editor';

function definition(): monaco.languages.IMonarchLanguage {
    return {
        keywords: [
            'AND',
            'ARRAY',
            'BEGIN',
            'BY',
            'CASE',
            'CONST',
            'DEFINITION',
            'DIV',
            'DO',
            'ELSE',
            'ELSIF',
            'END',
            'EXCEPT',
            'EXIT',
            'EXPORT',
            'FINALLY',
            'FOR',
            'FROM',
            'IF',
            'IMPLEMENTATION',
            'IMPORT',
            'IN',
            'LOOP',
            'MOD',
            'MODULE',
            'NOT',
            'OF',
            'OR',
            'PACKEDSET',
            'POINTER',
            'PROCEDURE',
            'QUALIFIED',
            'UNQUALIFIED',
            'RECORD',
            'REM',
            'REPEAT',
            'RETRY',
            'RETURN',
            'SET',
            'THEN',
            'TO',
            'TYPE',
            'UNTIL',
            'VAR',
            'WHILE',
            'WITH',
            'ASM',
            'VOLATILE',
        ],

        typeKeywords: ['integer number', 'real number', 'identifier', 'string'],

        numbers: /-?[0-9.]/,

        tokenizer: {
            root: [
                // identifiers and keywords
                [
                    /[A-Za-z_$][\w$]*/,
                    {
                        cases: {
                            '@typeKeywords': 'keyword',
                            '@keywords': 'keyword',
                            '@default': 'identifier',
                        },
                    },
                ],

                {include: '@whitespace'},

                [/@numbers/, 'number'],

                [/[+\-*/=<>$@]/, 'operators'],

                [/(")(.*)(")/, ['string', 'string', 'string']],
            ],

            comment: [
                [/[^(*]+/, 'comment'],
                [/\*\)/, 'comment', '@pop'],
            ],

            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/\(\*/, 'comment', '@comment'],
            ],
        },
    };
}

monaco.languages.register({id: 'modula2'});
monaco.languages.setMonarchTokensProvider('modula2', definition());

export {};
