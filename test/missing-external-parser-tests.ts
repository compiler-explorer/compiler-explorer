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

import fs from 'node:fs';
import {setTimeout as delay} from 'node:timers/promises';

import {afterEach, beforeAll, beforeEach, describe, expect, it, vi} from 'vitest';

import {AppArguments} from '../lib/app.interfaces.js';
import {ExternalParserBase} from '../lib/external-parsers/base.js';
import {MissingExternalParserError} from '../lib/external-parsers/missing-external-parser-error.js';
import {PlainParser} from '../lib/external-parsers/plain.js';
import {CompileHandler, SetTestMode} from '../lib/handlers/compile.js';
import {logger} from '../lib/logger.js';
import {fakeProps} from '../lib/properties.js';
import {PreliminaryCompilerInfo} from '../types/compiler.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

SetTestMode();

const missingCeParser = '/usr/local/bin/asm-parser-does-not-exist';
const missingPlainParser = '/opt/plain-parser-does-not-exist';
const existingParser = process.execPath;

function errorMessages(): string[] {
    return vi.mocked(logger.error).mock.calls.map(call => String(call[0]));
}

function warnMessages(): string[] {
    return vi.mocked(logger.warn).mock.calls.map(call => String(call[0]));
}

function exitTimers(): number[] {
    return vi.mocked(global.setTimeout).mock.calls.flatMap(call => {
        const callback = call[0];
        if (typeof callback !== 'function') return [];
        const source = callback.toString();
        // Vitest rewrites `process.exit` to a namespace import in the callback source.
        if (source.includes('process.exit') || source.includes('.exit(1)')) {
            return [Number(call[1])];
        }
        return [];
    });
}

const processExitSignal = new Error('process.exit');

function trackSettlement(pending: Promise<unknown>) {
    return pending.then(
        value => ({settled: 'resolved' as const, value}),
        (err: unknown) => ({settled: 'rejected' as const, err}),
    );
}

async function until(predicate: () => boolean) {
    for (let attempt = 0; attempt < 50; attempt++) {
        if (predicate()) return;
        await delay(1);
    }
    throw new Error('missing-parser flush point was not reached');
}

async function expectFlushThenExit(pending: Promise<unknown>) {
    const outcome = trackSettlement(pending);
    await until(() => errorMessages().some(message => message.includes('will not listen')));
    expect(process.exit).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(5000);
    expect(process.exit).toHaveBeenCalledTimes(1);
    expect(process.exit).toHaveBeenCalledWith(1);
    await expect(outcome).resolves.toEqual({settled: 'rejected', err: processExitSignal});
}

function missingParserCompiler(id: string, parserId: string, parserPath: string): PreliminaryCompilerInfo {
    return makeFakeCompilerInfo({
        id,
        lang: 'c++',
        exe: existingParser,
        compilerType: 'default',
        externalparser: {id: parserId, exe: parserPath, args: ''},
    }) as PreliminaryCompilerInfo;
}

function existingParserCompiler(id: string): PreliminaryCompilerInfo {
    return makeFakeCompilerInfo({
        id,
        lang: 'c++',
        exe: existingParser,
        compilerType: 'default',
        version: 'test-version',
        mtime: new Date(0),
        libsArr: [],
        externalparser: {id: 'CEAsmParser', exe: existingParser, args: ''},
    }) as PreliminaryCompilerInfo;
}

describe('External parser constructors', () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.spyOn(global, 'setTimeout');
        vi.spyOn(process, 'exit').mockImplementation(() => undefined as never);
    });

    afterEach(() => {
        vi.clearAllTimers();
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it('ExternalParserBase throws MissingExternalParserError and does not arm process.exit', () => {
        expect(fs.existsSync(missingCeParser)).toBe(false);
        const info = makeFakeCompilerInfo({
            id: 'gccrs',
            externalparser: {id: 'CEAsmParser', exe: missingCeParser, args: ''},
        });

        expect(() => new ExternalParserBase(info, {} as never, vi.fn() as never)).toThrow(MissingExternalParserError);
        try {
            new ExternalParserBase(info, {} as never, vi.fn() as never);
            expect.fail('constructor returned');
        } catch (err) {
            expect(err).toBeInstanceOf(MissingExternalParserError);
            expect((err as MissingExternalParserError).compilerId).toBe('gccrs');
            expect((err as MissingExternalParserError).parserPath).toBe(missingCeParser);
        }

        expect(exitTimers()).toEqual([]);
        vi.runAllTimers();
        expect(process.exit).not.toHaveBeenCalled();
    });

    it('PlainParser throws MissingExternalParserError and does not arm process.exit', () => {
        expect(fs.existsSync(missingPlainParser)).toBe(false);
        const info = makeFakeCompilerInfo({
            id: 'rustc',
            externalparser: {id: 'plain', exe: missingPlainParser, args: ''},
        });

        expect(() => new PlainParser(info, {} as never, vi.fn() as never)).toThrow(MissingExternalParserError);
        try {
            new PlainParser(info, {} as never, vi.fn() as never);
            expect.fail('constructor returned');
        } catch (err) {
            expect(err).toBeInstanceOf(MissingExternalParserError);
            expect((err as MissingExternalParserError).compilerId).toBe('rustc');
            expect((err as MissingExternalParserError).parserPath).toBe(missingPlainParser);
        }

        expect(exitTimers()).toEqual([]);
        vi.runAllTimers();
        expect(process.exit).not.toHaveBeenCalled();
    });
});

describe('CompileHandler missing external parser', () => {
    const appArgs = {devMode: false, exitOnCompilerFailure: false} as AppArguments;
    const clientOptions = {libs: {'c++': []}};
    let handler: CompileHandler;

    beforeAll(() => {
        handler = new CompileHandler(
            makeCompilationEnvironment({languages: {'c++': {id: 'c++'}}}),
            fakeProps({}),
            appArgs,
        );
    });

    beforeEach(() => {
        appArgs.devMode = false;
        appArgs.exitOnCompilerFailure = false;
        vi.useFakeTimers({toFake: ['setTimeout', 'clearTimeout']});
        vi.spyOn(global, 'setTimeout');
        vi.spyOn(process, 'exit').mockImplementation(() => {
            throw processExitSignal;
        });
        vi.spyOn(logger, 'error').mockImplementation(() => logger);
        vi.spyOn(logger, 'warn').mockImplementation(() => logger);
        vi.spyOn(logger, 'info').mockImplementation(() => logger);
        vi.spyOn(logger, 'debug').mockImplementation(() => logger);
    });

    afterEach(() => {
        vi.clearAllTimers();
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it('one missing parser outside devMode waits 5s then exits without returning', async () => {
        const pending = handler.setCompilers(
            [missingParserCompiler('gccrs', 'CEAsmParser', missingCeParser)],
            clientOptions as never,
        );

        await expectFlushThenExit(pending);
        const summary = errorMessages().find(message => message.includes('gccrs'));
        expect(summary).toContain(missingCeParser);
        expect(errorMessages().some(message => message.includes('Exiting in 5s so log transports can flush'))).toBe(
            true,
        );
        expect(exitTimers()).toEqual([]);
    });

    it('two missing parsers wait once, then exit once, with one summary', async () => {
        const pending = handler.setCompilers(
            [
                missingParserCompiler('gccrs', 'CEAsmParser', missingCeParser),
                missingParserCompiler('rustc', 'plain', missingPlainParser),
            ],
            clientOptions as never,
        );

        await expectFlushThenExit(pending);
        const summaries = errorMessages().filter(
            message => message.includes('gccrs') || message.includes('rustc') || message.includes(missingCeParser),
        );
        expect(summaries).toHaveLength(1);
        expect(summaries[0]).toContain('gccrs');
        expect(summaries[0]).toContain('rustc');
        expect(summaries[0]).toContain(missingCeParser);
        expect(summaries[0]).toContain(missingPlainParser);
    });

    it('devMode does not wait or exit and reports the miss as non-fatal', async () => {
        appArgs.devMode = true;

        const created = await handler.setCompilers(
            [missingParserCompiler('gccrs', 'CEAsmParser', missingCeParser)],
            clientOptions as never,
        );

        expect(created.map(compiler => compiler.id)).not.toContain('gccrs');
        expect(process.exit).not.toHaveBeenCalled();
        expect(errorMessages().some(message => message.includes('non-fatal only because devMode is on'))).toBe(true);
        expect(errorMessages().some(message => message.includes('will not listen'))).toBe(false);
        await vi.advanceTimersByTimeAsync(5000);
        expect(process.exit).not.toHaveBeenCalled();
    });

    it('an existing parser binary creates the compiler and does not arm an exit', async () => {
        expect(fs.existsSync(existingParser)).toBe(true);

        const created = await handler.setCompilers([existingParserCompiler('clang-ok')], clientOptions as never);

        expect(created.map(compiler => compiler.id)).toContain('clang-ok');
        expect(errorMessages().some(message => message.includes('MissingExternalParserError'))).toBe(false);
        expect(errorMessages().some(message => message.includes('Missing external parser'))).toBe(false);
        expect(exitTimers()).toEqual([]);
        expect(process.exit).not.toHaveBeenCalled();
    });

    it('a missing compiler binary logs Unable to stat and does not arm a missing-parser exit', async () => {
        const created = await handler.setCompilers(
            [
                {
                    id: 'missing-gcc',
                    lang: 'c++',
                    exe: '/no/such/ce-compiler-binary',
                    compilerType: 'default',
                } as PreliminaryCompilerInfo,
            ],
            clientOptions as never,
        );

        expect(created).toEqual([]);
        expect(warnMessages().some(message => message.includes('Unable to stat missing-gcc compiler binary'))).toBe(
            true,
        );
        expect(exitTimers()).toEqual([]);
        expect(process.exit).not.toHaveBeenCalled();
        await vi.advanceTimersByTimeAsync(5000);
        expect(process.exit).not.toHaveBeenCalled();
    });

    it('a later setCompilers call resets the missing-parser list and does not exit again', async () => {
        const first = handler.setCompilers(
            [missingParserCompiler('gccrs', 'CEAsmParser', missingCeParser)],
            clientOptions as never,
        );
        await expectFlushThenExit(first);

        vi.mocked(logger.error).mockClear();
        const created = await handler.setCompilers([existingParserCompiler('clang-ok')], clientOptions as never);

        expect(created.map(compiler => compiler.id)).toContain('clang-ok');
        expect(process.exit).toHaveBeenCalledTimes(1);
        expect(errorMessages().some(message => message.includes('gccrs'))).toBe(false);
        expect(errorMessages().some(message => message.includes(missingCeParser))).toBe(false);
    });

    it('exitOnCompilerFailure does not add a second exit for the same missing parser', async () => {
        appArgs.exitOnCompilerFailure = true;

        const pending = handler.setCompilers(
            [missingParserCompiler('gccrs', 'CEAsmParser', missingCeParser)],
            clientOptions as never,
        );

        await expectFlushThenExit(pending);
        expect(errorMessages().some(message => message.includes('exitOnCompilerFailure'))).toBe(false);
    });
});
