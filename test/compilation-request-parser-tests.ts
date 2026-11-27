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

import {describe, expect, it} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {defaultMaxToolsPerRequest, parseTools} from '../lib/compilation/compilation-request-parser.js';
import {CompilationInfo} from '../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {ToolResult} from '../types/tool.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

describe('parseTools', () => {
    it('splits string arguments', () => {
        expect(parseTools([{id: 'a', args: '-x "y z"', stdin: ''}])).toEqual([
            {id: 'a', args: ['-x', 'y z'], stdin: ''},
        ]);
    });

    it('drops repeated entries and keeps at most the cap', () => {
        const repeated = Array.from({length: 40}, () => ({id: 'clangtidytrunk', args: '--checks=*', stdin: ''}));
        expect(parseTools(repeated)).toEqual([{id: 'clangtidytrunk', args: ['--checks=*'], stdin: ''}]);

        const distinct = Array.from({length: 40}, (_, i) => ({id: 'clangtidytrunk', args: [`-${i}`], stdin: ''}));
        const parsed = parseTools(distinct);
        expect(parsed).toHaveLength(defaultMaxToolsPerRequest);
        expect(parsed.map(tool => tool.args)).toEqual([['-0'], ['-1'], ['-2'], ['-3'], ['-4']]);
    });

    it('treats the same tool with different arguments or stdin as different', () => {
        const tools = [
            {id: 'a', args: ['-1'], stdin: ''},
            {id: 'a', args: '-1', stdin: ''},
            {id: 'a', args: ['-2'], stdin: ''},
            {id: 'a', args: ['-1'], stdin: 'in'},
            {id: 'b', args: ['-1'], stdin: ''},
        ];
        expect(parseTools(tools).map(tool => [tool.id, tool.args, tool.stdin])).toEqual([
            ['a', ['-1'], ''],
            ['a', ['-2'], ''],
            ['a', ['-1'], 'in'],
            ['b', ['-1'], ''],
        ]);
    });

    it('honours an explicit cap', () => {
        const tools = Array.from({length: 10}, (_, i) => ({id: `t${i}`, args: [], stdin: ''}));
        expect(parseTools(tools, 3).map(tool => tool.id)).toEqual(['t0', 't1', 't2']);
    });
});

describe('runToolsOfType', () => {
    const languages = {'c++': {id: 'c++'}} as const;

    function makeCompiler(props: Record<string, any> = {}) {
        const ce = makeCompilationEnvironment({languages, props});
        const info = makeFakeCompilerInfo({exe: '', lang: 'c++', ldPath: []});
        return new BaseCompiler(info as CompilerInfo, ce);
    }

    it('limits how many tools run at once and returns results in request order', async () => {
        const compiler = makeCompiler();
        let running = 0;
        let maxRunning = 0;
        const fakeTool = {
            id: 'slow',
            type: 'independent',
            runTool: async (_info: CompilationInfo, _input?: string, args?: string[]): Promise<ToolResult> => {
                running++;
                maxRunning = Math.max(maxRunning, running);
                // Finish in reverse order, so request order can't come for free.
                await new Promise(resolve => setTimeout(resolve, 20 - Number(args?.[0] ?? 0)));
                running--;
                return {id: `slow-${args?.[0]}`, code: 0, stdout: [], stderr: []};
            },
        };
        (compiler as any).possibleTools = [fakeTool];

        const tools = parseTools(Array.from({length: 40}, (_, i) => ({id: 'slow', args: [`${i}`], stdin: ''})));
        const results = await Promise.all(
            compiler.runToolsOfType(tools, 'independent', {inputFilename: 'x'} as CompilationInfo),
        );

        expect(results.map(result => result.id)).toEqual(['slow-0', 'slow-1', 'slow-2', 'slow-3', 'slow-4']);
        expect(maxRunning).toBe(2);
    });

    it('reads its limits from properties', () => {
        const compiler = makeCompiler({maxToolsPerRequest: 3});
        expect(compiler.getMaxToolsPerRequest()).toBe(3);
        expect(makeCompiler().getMaxToolsPerRequest()).toBe(defaultMaxToolsPerRequest);
    });
});
