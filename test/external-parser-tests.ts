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

import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {ExternalParserBase} from '../lib/external-parsers/base.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';

const OBJDUMP = 'Z:/compilers/mingw-w64/bin/objdump.exe';
const OBJDUMP_ARGS = ['-d', 'output.s', '-l', '--insn-width=16', '-C', '-M', 'intel'];
const FILTERS = {binary: true, labels: true, libraryCode: true};

describe('External asm parser', () => {
    let buildfolder: string;
    let parserExe: string;
    let platform: PropertyDescriptor | undefined;

    beforeEach(async () => {
        buildfolder = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-external-parser-'));
        // The parser refuses to start without its executable
        parserExe = path.join(buildfolder, 'asm-parser');
        await fs.writeFile(parserExe, '');
        platform = Object.getOwnPropertyDescriptor(process, 'platform');
    });

    afterEach(async () => {
        if (platform) Object.defineProperty(process, 'platform', platform);
        vi.restoreAllMocks();
        await fs.rm(buildfolder, {recursive: true, force: true});
    });

    function makeParser(stdout = '{"asm":[]}', code = 0) {
        const execFunc = vi.fn().mockResolvedValue({code, stdout, stderr: ''});
        const parser = new ExternalParserBase(
            {id: 'test', objdumper: OBJDUMP, externalparser: {id: 'CEAsmParser', exe: parserExe}} as CompilerInfo,
            {getEnv: () => ({})} as unknown as CompilationEnvironment,
            execFunc,
        );
        return {parser, execFunc};
    }

    function onPlatform(name: NodeJS.Platform) {
        Object.defineProperty(process, 'platform', {value: name});
    }

    it('pipes objdump into the parser with a starter script', async () => {
        onPlatform('linux');
        const {parser, execFunc} = makeParser();

        const result = await parser.objdumpAndParseAssembly(buildfolder, OBJDUMP_ARGS, FILTERS);

        expect(execFunc).toHaveBeenCalledWith('./dump-and-parse.sh', OBJDUMP_ARGS, expect.anything());
        const script = await fs.readFile(path.join(buildfolder, 'dump-and-parse.sh'), 'utf8');
        expect(script).toContain('$OBJDUMP "$@" | $ASMPARSER -plt -stdin -binary -unused_labels -library_functions');
        expect(result.externalParserUsed).toBe(true);
    });

    it('has the parser run objdump itself on Windows, without a script', async () => {
        onPlatform('win32');
        const {parser, execFunc} = makeParser();

        const result = await parser.objdumpAndParseAssembly(buildfolder, OBJDUMP_ARGS, FILTERS);

        expect(execFunc).toHaveBeenCalledWith(
            parserExe,
            ['-plt', '-binary', '-unused_labels', '-library_functions', '-objdump', OBJDUMP, '--', ...OBJDUMP_ARGS],
            expect.objectContaining({customCwd: buildfolder}),
        );
        await expect(fs.access(path.join(buildfolder, 'dump-and-parse.sh'))).rejects.toThrow();
        expect(result.externalParserUsed).toBe(true);
    });

    it('reports what went wrong when the parser fails', async () => {
        onPlatform('win32');
        const {parser} = makeParser('', 1);

        await expect(parser.objdumpAndParseAssembly(buildfolder, OBJDUMP_ARGS, FILTERS)).rejects.toThrow(
            'Internal error running asm parser',
        );
    });
});
