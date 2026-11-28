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
import {parseTools} from '../lib/compilation/compilation-request-parser.js';
import {CompilationInfo, LegacyCompatibleActiveTool} from '../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {ToolResult} from '../types/tool.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

describe('parseTools', () => {
    it('splits string arguments', () => {
        expect(parseTools([{id: 'a', args: '-x "y z"', stdin: ''}])).toEqual([
            {id: 'a', args: ['-x', 'y z'], stdin: ''},
        ]);
    });

    it('keeps identical entries only once', () => {
        const repeated = Array.from({length: 40}, () => ({id: 'clangtidytrunk', args: '--checks=*', stdin: ''}));
        expect(parseTools(repeated)).toEqual([{id: 'clangtidytrunk', args: ['--checks=*'], stdin: ''}]);
    });

    it('keeps every distinct entry, in order', () => {
        const distinct = Array.from({length: 40}, (_, i) => ({id: 'clangtidytrunk', args: [`-${i}`], stdin: ''}));
        expect(parseTools(distinct).map(tool => tool.args)).toEqual(distinct.map(tool => tool.args));
    });

    it('treats missing args or stdin as empty when spotting repeats', () => {
        const tools = [
            {id: 'a', args: '-1'} as LegacyCompatibleActiveTool,
            {id: 'a', args: ['-1'], stdin: ''},
            {id: 'b'} as LegacyCompatibleActiveTool,
            {id: 'b', args: [], stdin: ''},
        ];
        expect(parseTools(tools).map(tool => tool.id)).toEqual(['a', 'b']);
    });

    it('still rejects a tools value that is not a list', () => {
        expect(() => parseTools('clangtidytrunk' as any)).toThrow();
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
});

describe('runToolsOfType', () => {
    const languages = {'c++': {id: 'c++'}} as const;
    const compilationInfo = {inputFilename: 'x'} as CompilationInfo;

    function makeCompilerWithSlowTool(props: Record<string, any> = {}) {
        const ce = makeCompilationEnvironment({languages, props});
        const info = makeFakeCompilerInfo({exe: '', lang: 'c++', ldPath: []});
        const compiler = new BaseCompiler(info as CompilerInfo, ce);
        const stats = {calls: 0, running: 0, maxRunning: 0};
        const slowTool = {
            id: 'slow',
            type: 'independent',
            runTool: async (_info: CompilationInfo, _input?: string, args?: string[]): Promise<ToolResult> => {
                stats.calls++;
                stats.running++;
                stats.maxRunning = Math.max(stats.maxRunning, stats.running);
                // Later entries finish sooner, so request order can't come for free.
                await new Promise(resolve => setTimeout(resolve, 30 - Number(args?.[0] ?? 0)));
                stats.running--;
                return {id: `slow-${args?.[0]}`, code: 0, stdout: [], stderr: []};
            },
        };
        (compiler as any).possibleTools = [slowTool];
        return {compiler, stats};
    }

    it('runs every tool, a limited number at a time, returning results in request order', async () => {
        const {compiler, stats} = makeCompilerWithSlowTool();
        const tools = parseTools(Array.from({length: 12}, (_, i) => ({id: 'slow', args: [`${i}`], stdin: ''})));

        const results = await Promise.all(compiler.runToolsOfType(tools, 'independent', compilationInfo));

        expect(results.map(result => result.id)).toEqual(Array.from({length: 12}, (_, i) => `slow-${i}`));
        expect(stats.calls).toBe(12);
        expect(stats.maxRunning).toBe(2);
    });

    it('falls back to the default limit for a property value that is not a positive integer', async () => {
        const {compiler, stats} = makeCompilerWithSlowTool({toolConcurrencyPerRequest: 'two'});
        const tools = parseTools(Array.from({length: 6}, (_, i) => ({id: 'slow', args: [`${i}`], stdin: ''})));

        await Promise.all(compiler.runToolsOfType(tools, 'independent', compilationInfo));

        expect(stats.maxRunning).toBe(2);
    });

    it('reads the limit from properties', async () => {
        const {compiler, stats} = makeCompilerWithSlowTool({toolConcurrencyPerRequest: 3});
        const tools = parseTools(Array.from({length: 8}, (_, i) => ({id: 'slow', args: [`${i}`], stdin: ''})));

        await Promise.all(compiler.runToolsOfType(tools, 'independent', compilationInfo));

        expect(stats.maxRunning).toBe(3);
    });

    it('starts no more tools once one has failed', async () => {
        const {compiler, stats} = makeCompilerWithSlowTool();
        const failing = {
            id: 'failing',
            type: 'independent',
            runTool: async (): Promise<ToolResult> => {
                throw new Error('tool setup failed');
            },
        };
        (compiler as any).possibleTools.push(failing);
        const tools = parseTools([
            {id: 'failing', args: [], stdin: ''},
            ...Array.from({length: 6}, (_, i) => ({id: 'slow', args: [`${i}`], stdin: ''})),
        ]);

        const results = compiler.runToolsOfType(tools, 'independent', compilationInfo);
        await expect(Promise.all(results)).rejects.toThrow('tool setup failed');
        await Promise.allSettled(results);

        // Only the tool that started alongside the failing one ran.
        expect(stats.calls).toBe(1);
    });

    it('runs identical entries once', async () => {
        const {compiler, stats} = makeCompilerWithSlowTool();
        const tools = parseTools(Array.from({length: 40}, () => ({id: 'slow', args: ['1'], stdin: ''})));

        const results = await Promise.all(compiler.runToolsOfType(tools, 'independent', compilationInfo));

        expect(results.map(result => result.id)).toEqual(['slow-1']);
        expect(stats.calls).toBe(1);
    });
});
