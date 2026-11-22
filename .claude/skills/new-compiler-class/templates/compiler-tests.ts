// Template for test/<key>-tests.ts. Copy the licence header from any file in test/, then replace
// MyCompiler, my-compiler and the expected command lines with the ones your class should produce.
// mylang must be a real key from lib/languages.ts: the constructor throws on an unknown language.
// optionsForFilter is protected in BaseCompiler: declare your override without `protected` so the test can call it.

import {beforeAll, describe, expect, it} from 'vitest';

import {MyCompiler} from '../lib/compilers/my-compiler.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    mylang: {id: 'mylang' as LanguageKey},
};

describe('MyCompiler', () => {
    let compiler: MyCompiler;

    beforeAll(() => {
        const ce = makeCompilationEnvironment({languages});
        compiler = new MyCompiler(makeFakeCompilerInfo({exe: '/dev/null', lang: languages.mylang.id}), ce);
    });

    it('emits assembly by default', () => {
        expect(compiler.optionsForFilter({}, 'output.s', [])).toEqual(['--emit-asm', '-o', 'output.s']);
    });

    it('links a binary when asked to', () => {
        expect(compiler.optionsForFilter({binary: true}, 'output.s', [])).toEqual(['-o', 'output.s']);
    });

    it('builds an object file for binaryObject', () => {
        expect(compiler.optionsForFilter({binaryObject: true}, 'output.s', [])).toEqual(['-c', '-o', 'output.s']);
    });
});
