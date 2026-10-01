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

type RecordedMissingParser = {id: string; parserPath: string};

function recordedMissingParsers(handler: CompileHandler): RecordedMissingParser[] {
    return (handler as unknown as {missingExternalParsers: RecordedMissingParser[]}).missingExternalParsers;
}

function errorMessages(): string[] {
    return vi.mocked(logger.error).mock.calls.map(call => String(call[0]));
}

function parserCompiler(id: string, parserId: string, parserPath: string): PreliminaryCompilerInfo {
    return makeFakeCompilerInfo({
        id,
        lang: 'c++',
        exe: process.execPath,
        compilerType: 'default',
        externalparser: {id: parserId, exe: parserPath, args: ''},
    }) as PreliminaryCompilerInfo;
}

describe('External parser constructors', () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.spyOn(process, 'exit').mockImplementation(() => undefined as never);
    });

    afterEach(() => {
        vi.clearAllTimers();
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it('throws MissingExternalParserError and does not schedule process.exit', () => {
        expect(fs.existsSync(missingCeParser)).toBe(false);
        expect(fs.existsSync(missingPlainParser)).toBe(false);

        const ceInfo = makeFakeCompilerInfo({
            id: 'gccrs',
            externalparser: {id: 'CEAsmParser', exe: missingCeParser, args: ''},
        });
        const plainInfo = makeFakeCompilerInfo({
            id: 'rustc',
            externalparser: {id: 'plain', exe: missingPlainParser, args: ''},
        });

        expect(() => new ExternalParserBase(ceInfo, {} as never, vi.fn() as never)).toThrow(MissingExternalParserError);
        expect(() => new PlainParser(plainInfo, {} as never, vi.fn() as never)).toThrow(MissingExternalParserError);

        try {
            new ExternalParserBase(ceInfo, {} as never, vi.fn() as never);
        } catch (err) {
            expect(err).toBeInstanceOf(MissingExternalParserError);
            expect((err as MissingExternalParserError).compilerId).toBe('gccrs');
            expect((err as MissingExternalParserError).parserPath).toBe(missingCeParser);
        }

        vi.runAllTimers();
        expect(process.exit).not.toHaveBeenCalled();
    });

    it('constructs when the parser binary exists', () => {
        const info = makeFakeCompilerInfo({
            id: 'gccrs',
            externalparser: {id: 'plain', exe: process.execPath, args: ''},
        });
        expect(() => new PlainParser(info, {} as never, vi.fn() as never)).not.toThrow();
        expect(() => new ExternalParserBase(info, {} as never, vi.fn() as never)).not.toThrow();
        vi.runAllTimers();
        expect(process.exit).not.toHaveBeenCalled();
    });
});

describe('CompileHandler missing external parser', () => {
    const appArgs = {devMode: false, exitOnCompilerFailure: false} as AppArguments;
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
        recordedMissingParsers(handler).length = 0;
        vi.useFakeTimers();
        vi.spyOn(process, 'exit').mockImplementation(() => undefined as never);
        vi.spyOn(logger, 'error').mockImplementation(() => logger);
        vi.spyOn(logger, 'warn').mockImplementation(() => logger);
        vi.spyOn(logger, 'info').mockImplementation(() => logger);
    });

    afterEach(() => {
        vi.clearAllTimers();
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it('records a missing parser from create and does not log Unable to stat', async () => {
        const created = await handler.create(parserCompiler('gccrs', 'CEAsmParser', missingCeParser));

        expect(created).toBeNull();
        expect(recordedMissingParsers(handler)).toEqual([{id: 'gccrs', parserPath: missingCeParser}]);
        const warnings = vi.mocked(logger.warn).mock.calls.map(call => String(call[0]));
        expect(warnings.some(message => message.includes('Unable to stat'))).toBe(false);
        vi.runAllTimers();
        expect(process.exit).not.toHaveBeenCalled();
    });

    it('still logs Unable to stat when the compiler binary itself is missing', async () => {
        const created = await handler.create({
            id: 'missing-gcc',
            lang: 'c++',
            exe: '/no/such/ce-compiler-binary',
            compilerType: 'default',
        } as PreliminaryCompilerInfo);

        expect(created).toBeNull();
        expect(recordedMissingParsers(handler)).toEqual([]);
        const warnings = vi.mocked(logger.warn).mock.calls.map(call => String(call[0]));
        expect(warnings.some(message => message.includes('Unable to stat missing-gcc compiler binary'))).toBe(true);
    });

    it('arms a single exit outside devMode and the log names every compiler id and parser path', async () => {
        const created = await handler.setCompilers(
            [
                parserCompiler('gccrs', 'CEAsmParser', missingCeParser),
                parserCompiler('rustc', 'plain', missingPlainParser),
            ],
            null as never,
        );

        expect(created).toEqual([]);
        expect(process.exit).not.toHaveBeenCalled();

        const errors = errorMessages();
        const summary = errors.find(message => message.includes('gccrs') && message.includes('rustc'));
        expect(summary).toBeDefined();
        expect(summary).toContain(missingCeParser);
        expect(summary).toContain(missingPlainParser);
        expect(errors.filter(message => message.includes(missingCeParser)).length).toBe(1);
        expect(errors.some(message => message.includes('Exiting in 5s so log transports can flush'))).toBe(true);

        await vi.advanceTimersByTimeAsync(4999);
        expect(process.exit).not.toHaveBeenCalled();
        await vi.advanceTimersByTimeAsync(1);
        expect(process.exit).toHaveBeenCalledTimes(1);
        expect(process.exit).toHaveBeenCalledWith(1);
    });

    it('does not exit in devMode', async () => {
        appArgs.devMode = true;

        await handler.setCompilers([parserCompiler('gccrs', 'CEAsmParser', missingCeParser)], null as never);

        const errors = errorMessages();
        expect(errors.some(message => message.includes('gccrs') && message.includes(missingCeParser))).toBe(true);
        expect(errors.some(message => message.includes('non-fatal only because devMode is on'))).toBe(true);
        await vi.runAllTimersAsync();
        expect(process.exit).not.toHaveBeenCalled();
    });

    it('does not also take the immediate exitOnCompilerFailure path for a missing parser', async () => {
        appArgs.exitOnCompilerFailure = true;

        await handler.setCompilers([parserCompiler('gccrs', 'CEAsmParser', missingCeParser)], null as never);

        expect(process.exit).not.toHaveBeenCalled();
        await vi.advanceTimersByTimeAsync(5000);
        expect(process.exit).toHaveBeenCalledTimes(1);
        expect(process.exit).toHaveBeenCalledWith(1);
    });
});
