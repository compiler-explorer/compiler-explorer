// Copyright (c) 2025, Compiler Explorer Authors
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

import {beforeAll, describe, expect, it} from 'vitest';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {ClojureCompiler} from '../lib/compilers/index.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';

import {makeCompilationEnvironment} from './utils.js';

const languages = {
    clojure: {id: 'clojure'},
};

const info = {
    exe: '/opt/clojure/test-version/bin/clojure',
    remote: true,
    lang: languages.clojure.id,
} as unknown as CompilerInfo;

describe('Basic compiler setup', () => {
    let env: CompilationEnvironment;

    beforeAll(() => {
        env = makeCompilationEnvironment({languages});
    });

    it('Should not crash on instantiation', () => {
        new ClojureCompiler(info, env);
    });

    describe('Forbidden compiler arguments', () => {
        it('ClojureCompiler should not allow a parameter with .clj suffix', () => {
            // The ClojureCompiler filters out any option that looks like a source file.
            // Other invalid options are filtered out by the wrapper.
            const compiler = new ClojureCompiler(info, env);
            expect(compiler.filterUserOptions(['hello', 'hello.clj', '-d'])).toEqual(['hello', '-d']);
        });

        it('ClojureCompiler should not allow user to set --macro-expand parameter', () => {
            // --macro-expand is used internally to produce macro expanded output.
            const compiler = new ClojureCompiler(info, env);
            expect(compiler.filterUserOptions(['--macro-expand', '--something-else'])).toEqual(['--something-else']);
        });
    });
});
