// Copyright (c) 2026, Compiler Explorer Authors
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

import path from 'node:path';

import {beforeAll, describe, expect, it} from 'vitest';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {LuaCompiler} from '../lib/compilers/lua.js';
import {languages} from '../lib/languages.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

// Subclass that exposes protected hooks so the tests can validate the
// extension points alternative Lua implementations would use.
class TestLuaCompiler extends LuaCompiler {
    public callResolveLuacExe(): string {
        return this.resolveLuacExe();
    }

    public callGetDisassemblyArgs(userOptions: string[], inputFilename: string): string[] {
        return this.getDisassemblyArgs(userOptions, inputFilename);
    }
}

const testLanguages = {
    lua: {id: 'lua' as LanguageKey},
};

describe('Lua language definition', () => {
    it('is registered with the expected metadata', () => {
        expect(languages.lua).toBeDefined();
        expect(languages.lua.id).toBe('lua');
        expect(languages.lua.name).toBe('Lua');
        expect(languages.lua.monaco).toBe('lua');
        expect(languages.lua.extensions[0]).toBe('.lua');
    });

    it('has a default example', () => {
        expect(languages.lua.example).not.toMatch(/something went wrong/i);
        expect(languages.lua.example.length).toBeGreaterThan(0);
    });
});

describe('LuaCompiler', () => {
    let ce: CompilationEnvironment;
    const luaExe = path.join('/', 'opt', 'lua-5.4.7', 'bin', 'lua5.4');
    const info = {
        exe: luaExe,
        remote: {
            target: 'foo',
            path: 'bar',
            cmakePath: 'cmake',
            basePath: '/',
        },
        lang: testLanguages.lua.id,
    };

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages: testLanguages});
    });

    it('uses BaseParser for argument parsing', () => {
        const compiler = new TestLuaCompiler(makeFakeCompilerInfo(info), ce);
        const parser = compiler.getArgumentParserClass();
        expect(parser.name).toBe('BaseParser');
    });

    it('emits no framework options for the lua filter', () => {
        const compiler = new TestLuaCompiler(makeFakeCompilerInfo(info), ce);
        expect(compiler.optionsForFilter({} as any, '/tmp/output')).toEqual([]);
    });

    it('derives luac path from the configured lua binary by default', () => {
        const compiler = new TestLuaCompiler(makeFakeCompilerInfo(info), ce);
        expect(compiler.callResolveLuacExe()).toBe(path.join('/opt', 'lua-5.4.7', 'bin', 'luac5.4'));
    });

    it('derives luac path correctly when exe is just "lua"', () => {
        const plainInfo = {...info, exe: '/usr/bin/lua'};
        const compiler = new TestLuaCompiler(makeFakeCompilerInfo(plainInfo), ce);
        expect(compiler.callResolveLuacExe()).toBe(path.join('/usr/bin', 'luac'));
    });

    it('builds a luac listing command line for disassembly', () => {
        const compiler = new TestLuaCompiler(makeFakeCompilerInfo(info), ce);
        expect(compiler.callGetDisassemblyArgs([], '/tmp/input.lua')).toEqual(['-l', '-l', '-p', '/tmp/input.lua']);
    });

    it('processAsm parses source line numbers from luac listing format', async () => {
        const compiler = new TestLuaCompiler(makeFakeCompilerInfo(info), ce);
        const sample = [
            'main <example.lua:0,0> (5 instructions at 0x1000)',
            '0+ params, 2 slots, 1 upvalue, 0 locals, 1 constant, 1 function',
            '\t1\t[3]\tVARARGPREP\t0',
            '\t2\t[1]\tCLOSURE  \t0 0\t; 0x1100',
            '\t3\t[3]\tSETTABUP \t0 0 0\t; _ENV "square"',
            '\t4\t[3]\tRETURN   \t0 1 1\t; 0 out',
            '',
            'function <example.lua:1,3> (3 instructions at 0x1100)',
            '\t1\t[2]\tMUL      \t1 0 0',
            '\t2\t[2]\tRETURN1  \t1',
        ].join('\n');

        const {asm} = await compiler.processAsm({asm: sample});

        const sourceLines = asm.map(line => (line.source ? line.source.line : null));
        // The instruction lines must report the source line in their `[N]` token.
        expect(sourceLines).toContain(3);
        expect(sourceLines).toContain(1);
        expect(sourceLines).toContain(2);

        // Each input line is preserved as text in order.
        expect(asm.length).toBe(sample.split('\n').length);
        expect(asm[0].text).toBe(sample.split('\n')[0]);
    });
});
