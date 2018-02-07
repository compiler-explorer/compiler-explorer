// Copyright (c) 2012-2018, Matt Godbolt, Rubén Rincón, byteally
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


"use strict";

// From https://github.com/byteally/monaco-haskell
// It does not work right now. Here as a reminder until the author fixes it
function definition() {
    return {
        module: /^(module)\b/,
        import: /^(import)\b/,
        module_name: /([A-Z][\w']*)(\.[A-Z][\w']*)*/,
        module_exports: /([A-Z][\w']*)(\.[A-Z][\w']*)*/,
        target: /([A-Z][\w']*)(\.[A-Z][\w']*)*/,
        invalid: /[a-z]+/,
        datatype: /[A-Z][\w]+/,
        datatypekey: /Int|Float|Text/,
        functions: /[a-z]+/,
        datacons: /[A-Z][\w]+/,
        typcons: /\s*[A-Z][\w]+/,
        text: /[A-Z][\w]+/,
        classname: /\s*[A-Z][\w]+/,
        arguments: /[a-z][\w]*/,
        reservedid: /qualified|hiding|case|default|deriving|do|else|if|import|in|infix|infixr|let|of|then|type|where|show|_/,
        tokenizer: {
            root: [
                [/(@module)/, 'keyword.module.haskell', '@module'],
                [/@import/, 'keyword.module.haskell', '@import'],
                [/\bdata\b(?=@typcons)/, 'keyword', '@typcons'],
                [/\bnewtype\b(?=\s*@typcons)/, 'keyword', '@typcons'],
                [/\bclass\b(?=\s*@classname)/, 'keyword', '@classname'],
                [/\bclass\b(?=\s*\()/, 'keyword', '@classname'],
                [/@reservedid/, 'keyword'],
                [/[a-z][\w]+(?=\s*?=)/, 'binderas', '@binder'],
                [/[a-z][\w]+(?=\s*::)/, 'binderad', '@typconss'],
                [/[a-z][\w]+(?=\s*?[a-z]\w*)/, 'binder', '@typconss'],
                [/[a-z][\w]+(?=\s*?::\s*?@datatypekey)/, 'bind', '@binder'],
                [/[a-z][\w]+(?=\s*?=\s*?@datacons)/, 'bindings'],
                {include: '@whitespace'},
                {include: '@comment'}
            ],
            import: [
                [/@module_name/, 'storage.module,haskell'],
                [/(@module_name)(?=\s*as)/, 'storage.module.haskell'],
                [/\b(qualified)\b(?=\s*@module_name)/, 'keyword'],
                [/(@module_name)(?=\s*?\()/, 'storage.module.haskell'],
                [/(@module_name)(?=\s*\bhiding\b)/, 'storage.module.haskell'],
                [/\b(hiding)\b(?=\s*?\()/, 'keyword'],
                [/\b(as)\b(?=\s*@module_name)/, 'keyword'],
                [/\(/, 'openbracketss', '@functions'],
                {include: '@comment'}
                //{include:'@blockComment'}
            ],
            module: [
                [/(@module_name)/, 'storage.module.haskell'],
                [/\(/, 'openbracket', '@module_exports'],
                [/@reservedid/, 'keyword', '@pop'],
                [/@invalid/, 'invalid'],
                [/,/, 'commas'],
                {include: '@whitespace'},
                {include: '@comment'}
                //{include:'@blockComment'}
            ],
            functions: [
                [/@datatype/, 'dcataype', '@datatype'],
                [/@functions/, 'functions'],
                [/,/, 'commas'],
                [/\)(?=\))/, 'closebracket'],
                [/\)/, 'closebracketalla', '@popall'],
                {include: '@comment'}
                //{include:'@blockComment'}
            ],
            typconss: [
                [/=/, 'eqals'],
                [/\bundefined\b/, 'val', '@all'],
                [/\bclass\b(?=\s*@classname)/, 'keyword', '@classname'],
                [/\bclass\b(?=\s*\()/, 'keyword', '@classname'],
                [/\binstance\b/, 'keyword', '@classname'],
                [/\bdata\b(?=@typcons)/, 'keyword', '@typcons'],
                [/\bnewtype\b(?=\s*@typcons)/, 'keyword', '@typcons'],
                [/@reservedid/, 'keyword', '@function'],
                [/[a-z]\w*/, 'arguments121'],
                [/[a-z][\w]+(?=\s*=)/, 'binderad', '@datacons'],
                [/[a-z][\w]+(?=\s*::)/, 'binderad', '@typconss'],
                [/[a-z][\w]+(?=\s*[a-z]\w*)/, 'binderad', '@typconss'],
                [/[a-z]\w*/, 'arguments121'],
                [/::/, 'dcol', '@datacons'],
                [/->/, 'pipes', '@binder'],
                {include: '@whitespace'},
                {include: '@comment'}
                //{include:'@blockComment'}
            ],

            function: [
                [/@reservedid/, 'keyword'],
                [/@datatypekey/, 'dtype'],
                [/[a-z][\w]+(?=\s*::)/, 'binderad', '@typconss'],
                [/[a-z][\w]+(?=\s*?[a-z]\w*)/, 'binder', '@typconss'],
                [/[a-z]\w*/, 'arguments12'],
                [/::/, 'dcol'],
                [/->/, 'pipe'],
                [/\bclass\b(?=\s*@classname)/, 'keyword', '@classname'],
                [/\bclass\b(?=\s*\()/, 'keyword', '@classname'],
                [/\binstance\b/, 'keyword', '@classname'],
                [/\bdata\b(?=@typcons)/, 'keyword', '@typcons'],
                [/\bnewtype\b(?=\s*@typcons)/, 'keyword'],
                {include: '@comment'}
                //{include:'@blockComment'}
            ],
            typcons: [
                [/=>/, 'pipes'],
                [/->/, 'pipe', '@bind'],
                [/\bclass\b(?=\s*@classname)/, 'keyword', '@classname'],
                [/\bclass\b(?=\s*\()/, 'keyword', '@classname'],
                [/\binstance\b/, 'keyword', '@classname'],
                [/\bdata\b(?=@typcons)/, 'keyword', '@typcons'],
                [/\bnewtype\b(?=\s*@typcons)/, 'keyword'],
                [/@reservedid(?=\s*\()/, 'keyword', '@typcons'],
                [/@reservedid(?=\s*[A-Z][\w]+)/, 'keyword', '@typcons'],
                [/@reservedid/, 'keyword', '@binder'],
                [/@datatypekey/, 'dtypes2'],
                [/[A-Z][\w]+(?=\s*,)/, 'classname', '@types'],
                [/[A-Z][\w]+(?=\s*[a-z][\w]*\s*,)/, 'classname', '@types'],
                [/@arguments(?=\s*,\s*)/, 'argument', '@types'],

                [/[A-Z][\w]+/, 'typecons'],
                [/[a-z][\w]+(?=\s*?=)/, 'binderas', '@binder'],
                [/[a-z][\w]+(?=\s*::)/, 'binderad', '@typconss'],
                [/[a-z][\w]+(?=\s*?[a-z]\w*)/, 'bind', '@typconss'],
                [/[a-z][\w]+(?=\s*?::\s*?@datatypekey)/, 'bind', '@binder'],
                [/[a-z][\w]+(?=\s*?=\s*?@datacons)/, 'bindings'],
                [/{/, 'oopen', '@initialise'],
                [/[a-z]\w*/, 'arguments13'],

                [/\(/, 'open_bracket'],
                [/=/, 'Equalss', '@datacons'],
                [/,/, 'commaa'],
                [/::/, 'colondouble', '@datacons'],
                [/\)/, 'closebrak', '@initialise'],
                [/\)(?=\s* \t\r\n\bdata\b)/, 'closed', '@popall'],
                {include: '@whitespace'},
                {include: '@comment'}
//                  {include:'@blockComment'}
            ],
            bind: [
                [/@datatypekey/, 'dtyp', '@typconss'],
                [/[a-z][\w]*/, 'typvar', '@typconss'],
                [/\(/, 'opens', '@typcons'],
                {include: '@whitespace'}
            ],
            types: [
                [/[a-z][\w]*/, 'argument123'],
                [/,/, 'commaa'],
                [/[A-Z][\w]*/, 'classname'],
                [/\)/, 'closex'],
                [/=>/, 'pipe', '@type'],
                {include: '@whitespace'},
                {include: '@comment'}
//                  {include:'@blockComment'}
            ],
            datacons: [
                [/=/, 'Equalst'],
                [/\bclass\b(?=\s*@classname)/, 'keyword', '@classname'],
                [/\bclass\b(?=\s*\()/, 'keyword', '@classname'],

                [/\bundefined\b/, 'val', '@all'],
                [/\bnewtype\b(?=\s*@typcons)/, 'keyword', '@typcons'],
                [/\binstance\b/, 'keyword', '@classname'],
                [/[a-z][\w]*(?=\s*?->)/, 'typvar', '@typcons'],

                [/@reservedid(?=\s*\()/, 'keyword', '@typcons'],
                [/@datatypekey(?=\s*[a-z]\w*)/, 'dtypess1', '@typcons'],
                [/@datatypekey/, 'dtypes1'],
                [/@reservedid(?=\s*[A-Z][\w]+)/, 'keyword', '@typcons'],
                [/@reservedid/, 'keyword'],
                [/[A-Z][\w]+(?=\s*=>)/, 'classname', '@type'],
                [/[A-Z][\w]+(?=\s*[a-z][\w]*\s*=>)/, 'classnames', '@type'],
                [/[a-z][\w]+(?=\s*?::)/, 'bindera', '@typconss'],
                [/[a-z][\w]+(?=\s*?[a-z]\w*)/, 'binde', '@typconss'],

                [/[a-z][\w]+/, 'binder', '@typconss'],

                [/\bdata\b(?=@typcons)/, 'keyword', '@typcons'],

                [/@datacons/, 'datacon'],
                [/@reservedid/, 'keyword', '@binder'],
                [/@datacons(?=\s*@datatypekey)/, 'datacons'],

                [/[a-z][\w]+(?=\s*?::\s*@datatype,)/, 'bindera', '@binder'],
                [/[a-z][\w]+(?=\s*?::\s*@datatype,)/, 'bindera', '@binder'],
                [/[a-z][\w]+(?=\s*?=\s*(\d)+,)/, 'binderb', '@binder'],
                [/[a-z][\w]+(?=\s*?=\s*(\d)+)/, 'binderc', '@binder'],
                [/[a-z][\w]+(?=\s*?=\s*"(\w)+.+?")/, 'binderd', '@binder'],
                [/[a-z][\w]+(?=\s*?=\s*"(\w)+.+?",)/, 'bindere', '@binder'],
                [/{/, 'openbracketd', '@initialise'],
                [/"\w+.+"/, 'type'],
                [/\d+/, 'type'],
                [/->/, 'pip', '@binder'],
                [/,/, 'commas33', '@initialise'],
                [/\|/, 'alternate'],
                [/\(/, 'open', '@typcons'],
                [/}/, 'clos', '@initialise'],
                {include: '@whitespace'},
                {include: '@comment'}
//                  {include:'@blockComment'}
            ],
            type: [
                [/[a-z][\w]*/, 'argument'],
                [/,/, 'commas'],
                [/=>/, 'pipe'],
                [/[A-Z][\w]*/, 'typcons'],
                [/->/, 'pipe', '@binder'],
                {include: '@whitespace'},
                {include: '@comment'}
//                  {include:'@blockComment'}
            ],
            all: [
                [/=/, 'equals'],

                [/\binstance\b/, 'keyword', '@classname'],
                [/\bclass\b(?=\s*@classname)/, 'keyword', '@classname'],
                [/\bclass\b(?=\s*\()/, 'keyword', '@classname'],
                [/\bdata\b(?=@typcons)/, 'keyword', '@typcons'],

                [/\bnewtype\b(?=\s*@typcons)/, 'keyword'],
                [/@reservedid/, 'keyword', '@typconss'],
                [/[A-Z][\w]+/, 'typecons', '@typcons'],
                [/[a-z][\w]+(?=\s*=)/, 'binderad', '@typconss'],
                [/[a-z][\w]+(?=\s*::)/, 'binderad', '@typconss'],
                [/[a-z][\w]+(?=\s*?[a-z]\w*)/, 'binder', '@typconss'],
                {include: '@comment'}
//                  {include:'@blockComment'}
            ],
            selectors: [
                [/[a-z][\w]+/, 'selectors'],
                [/=/, 'Equalsto', '@datacons'],
                [/"(\w)+.+"/, 'type'],
                [/\d+/, 'type'],
                [/}/, 'closeS'],
                {include: '@comment'}
//                  {include:'@blockComment'}
            ],
            initialise: [
                [/\binstance\b/, 'keyword', '@classname'],
                [/\bdata\b(?=\s*@typcons)/, 'keyword', '@typcons'],
                [/\bnewtype\b(?=\s*@typcons)/, 'keyword', '@typcons'],
                [/\bclass\b(?=\s*@classname)/, 'keyword', '@classname'],
                [/\bclass\b(?=\s*\()/, 'keyword', '@classname'],
                [/@datatypekey(?=\s*)/, 'datatype'],
                [/{/, 'open'],
                [/@reservedid/, 'keyword'],
                [/\(/, 'opn', '@typcons'],
                [/\|/, 'alternates', '@datacons'],
                [/,/, 'comma3'],

                [/[a-z][\w]+(?=\s*?=\s*?\b@reservedid\b)/, 'binder', '@all'],
                [/[a-z][\w]+(?=\s*?=\s*?\bundefined\b)/, 'binder', '@binder'],
                [/[a-z][\w]+(?=\s*?=\s*?@datacons)/, 'bindings'],
                [/[a-z][\w]+(?=\s*?::\s*@datatypekey)/, 'bindera', '@binder'],
                [/[a-z][\w]+(?=\s*::)/, 'binder', '@binder'],

                [/[a-z][\w]+(?=\s*?::\s*@datatypekey,)/, 'binderb', '@binder'],
                [/[a-z][\w]+(?=\s*?=\s*(\d)+,)/, 'binderc', '@binder'],
                [/[a-z][\w]+(?=\s*?=\s*(\d)+)/, 'binderd', '@binder'],
                [/[a-z][\w]+(?=\s*?=\s*"(\w)+.+?",)/, 'binders', '@binder'],
                [/[a-z][\w]+(?=\s*?=\s*"(\w)+.+?")/, 'bindere', '@binder'],

                [/[a-z][\w]+(?=\s*[a-z]\w*)/, 'binder', '@typconss'],
                [/::/, 'doublecolons', '@datacons'],
                [/=/, 'Equalsto', '@datacons'],
                {include: '@whitespace'},
                {include: '@comment'}
//                  {include:'@blockComment'}
            ],
            classname: [
                [/\binstance\b/, 'keyword', '@classname'],
                [/\bclass\b(?=\s*@classname)/, 'keyword', '@classname'],
                [/\bclass\b(?=\s*\()/, 'keyword', '@classname'],
                [/\bdata\b(?=\s*?@typcons)/, 'keyword', '@typcons'],
                [/\bnewtype\b(?=\s*@typcons)/, 'keyword', '@typcons'],
                [/[a-z][\w]+(?=\s*?=)/, 'binderas', '@binder'],
                [/[a-z][\w]+(?=\s*?::\s*?@datatypekey)/, 'binderas', '@binder'],
                [/@datatypekey/, 'dtype'],
                [/[A-Z][\w]*/, 'classname'],
                [/@reservedid/, 'keyword', '@typcons'],
                [/[a-z]\w*/, 'arguments'],
                [/,/, 'commas'],
                [/\(/, 'openbracket'],
                [/\)/, 'closebracket'],
                [/=>/, 'pipe', '@superclass'],
                {include: '@whitespace'},
                {include: '@comment'}
//                  {include:'@blockComment'}
            ],
            binder: [
                [/\binstance\b/, 'keyword', '@classname'],
                [/\bclass\b(?=\s*@classname)/, 'keyword', '@classname'],
                [/\bclass\b(?=\s*\()/, 'keyword', '@classname'],
                [/\bdata\b(?=\s*?@typcons)/, 'keyword', '@typcons'],
                [/@reservedid/, 'keyword'],
                [/[a-z][\w]+(?=\s*::)/, 'binder'],
                [/[a-z][\w]+(?=\s*=)/, 'binder'],
                [/::/, 'doublecolons', '@datacons'],
                [/@datatypekey(?=\s*[a-z]\w*)/, 'dtypes', '@typcons'],
                [/@datatypekey/, 'dtypess'],
                [/\bnewtype\b(?=\s*@typcons)/, 'keyword', '@typcons'],
                [/->/, 'pipe'],
                [/\d+/, 'digit'],
                [/{/, 'open'],
                [/=/, 'Equalsto', '@datacons'],

                [/\(/, 'opn', '@typcons'],
                [/\|/, 'alternates', '@datacons'],
                [/,/, 'comma3'],

                [/[a-z][\w]*/, 'typvar', '@typconss'],
                [/\bclass\b(?=\s*@classname)/, 'keyword', '@classname'],
                [/\bclass\b(?=\s*\()/, 'keyword', '@classname'],
                [/\binstance\b/, 'keyword', '@classname'],
                [/\bundefined\b/, 'value'],
                {include: '@whitespace'},
                {include: '@comment'}
//                  {include:'@blockComment'}
            ],
            superclass: [
                [/[A-Z][\w]+/, 'typeclass'],
                [/@reservedid/, 'keyword', '@typcons'],
                [/[a-z][\w]*/, 'typearguments'],
                {include: '@whitespace'},
                {include: '@comment'}
                //{include:'@blockComment'}
            ],
            argument: [
                [/[a-z][\w]*(?=\s*->)/, 'argument'],
                [/->/, 'pipea', '@binder'],
                {include: '@whitespace'},
                {include: '@comment'}
//                  {include:'@blockComment'}
            ],
            datatype: [
                [/\(/, 'openbracketd', '@pop'],
                {include: '@comment'}
                //{include:'@blockComment'}
            ],
            module_exports: [
                [/(@module_exports)/, 'storage.module.haskell'],
                [/\(/, 'openbracket', '@target'],
                [/\)/, 'closebracket', '@popall'],
                [/,/, 'comma1'],
                [/@invalid/, 'invalid'],
                {include: '@whitespace'},
                {include: '@comment'}
//                  {include:'@blockcomment'}
            ],
            target: [
                [/(@target)/, 'target'],
                [/,/, 'comma2'],
                [/\)/, 'closebracket', '@pop'],
                {include: '@comment'}
//                  {include:'@blockComment'}
            ],
            whitespace: [
                [/\t\r\n/, 'whitespace'],
                [/\s*/, 'whitespace']

            ],
            comment: [
                [/--/, 'punctuation.comment.haskell'],
                {include: '@whitespace'}
            ],
            Block_comment: [
                [/{-/, 'punctuation.comment.haskell'],
                [/-}/, 'comment.block.haskell'],
                {include: '@whitespace'}
            ]
        }
    };
}

monaco.languages.register({id: 'haskell'});
// TODO: Currently not working. Here as a reminder
/*monaco.languages.setMonarchTokensProvider('haskell', definition());*/