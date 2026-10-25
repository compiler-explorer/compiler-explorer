// Copyright (c) 2017, Compiler Explorer Authors
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

import {describe, expect, it} from 'vitest';

import {MapFileReader} from '../lib/mapfiles/map-file.js';
import {MapFileReaderVS} from '../lib/mapfiles/map-file-vs.js';
import {PELabelReconstructor} from '../lib/pe32-support.js';

describe('Basic reconstructions', () => {
    it('No lines', () => {
        const lines: string[] = [];
        const reconstructor = new PELabelReconstructor(lines, false, new MapFileReader('unused'), false);
        expect(reconstructor.asmLines.length).toEqual(0);
    });
});

describe('Visual Studio reconstruction', () => {
    function reconstruct() {
        const asm = fs.readFileSync('test/maps/squareprintfstaticlib.asm', 'utf16le').split('\n');
        const reconstructor = new PELabelReconstructor(
            asm,
            false,
            new MapFileReaderVS('test/maps/squareprintfstaticlib.map'),
        );
        reconstructor.run('output.s.obj');
        return reconstructor.asmLines;
    }

    function sourceOf(lines: string[], label: string) {
        const idx = lines.findIndex(line => line.endsWith(`<${label}>:`));
        expect(idx).toBeGreaterThan(-1);
        const next = lines[idx + 1];
        return next.startsWith('/app/') ? next : null;
    }

    it("gives the program's own functions their object as source", () => {
        const lines = reconstruct();
        expect(sourceOf(lines, 'main')).toEqual('/app/output.s.obj:0');
        expect(sourceOf(lines, '?square@@YAHH@Z')).toEqual('/app/output.s.obj:0');
    });

    it('gives functions from library archives no source, although they share a section with the program', () => {
        const lines = reconstruct();
        expect(sourceOf(lines, 'mainCRTStartup')).toBeNull();
        expect(sourceOf(lines, '?getValue@String@Groundfloor@@QEBAPEADXZ')).toBeNull();
    });

    it('tells library and linker units from the program objects', () => {
        expect(PELabelReconstructor.isProgramUnit('output.s.obj')).toBe(true);
        expect(PELabelReconstructor.isProgramUnit('main.cpp.obj')).toBe(true);
        expect(PELabelReconstructor.isProgramUnit('msvcrt:exe_main.obj')).toBe(false);
        expect(PELabelReconstructor.isProgramUnit('* CIL library *:* CIL module *')).toBe(false);
        expect(PELabelReconstructor.isProgramUnit('<absolute>')).toBe(false);
    });
});
