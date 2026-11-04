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

import {beforeEach, describe, expect, it, vi} from 'vitest';

import * as exec from '../lib/exec.js';
import {FormattingService} from '../lib/formatting-service.js';
import {fakeProps} from '../lib/properties.js';

vi.mock('../lib/exec.js');

describe('Formatting service with clang-format', () => {
    let service: FormattingService;

    beforeEach(async () => {
        vi.mocked(exec.execute).mockResolvedValue({
            code: 0,
            okToCache: true,
            filenameTransform: (f: string) => f,
            stdout: 'formatted',
            stderr: '',
            execTime: 0,
            timedOut: false,
            truncated: false,
        });
        service = new FormattingService();
        await service.initialize(
            fakeProps({
                formatters: 'clangformat',
                'formatter.clangformat.exe': '/usr/bin/clang-format',
                'formatter.clangformat.type': 'clangformat',
                'formatter.clangformat.styles': 'MyCorp:Google',
                'formatter.clangformat.styleFile.MyCorp': '/etc/mycorp/.clang-format',
            }),
        );
        vi.mocked(exec.execute).mockClear();
    });

    it('should only record style files for configured styles', () => {
        const info = service.getFormatterById('clangformat')?.formatterInfo;
        expect(info?.styles).toEqual(['MyCorp', 'Google']);
        expect(info?.styleFiles).toEqual({MyCorp: '/etc/mycorp/.clang-format'});
    });

    it('should use the style file for a file-backed style', async () => {
        await service.format('clangformat', 'int x;', {baseStyle: 'MyCorp', tabWidth: 4, useSpaces: true});
        expect(exec.execute).toHaveBeenCalledWith('/usr/bin/clang-format', ['--style=file:/etc/mycorp/.clang-format'], {
            input: 'int x;',
        });
    });

    it('should build an inline style for a built-in style', async () => {
        await service.format('clangformat', 'int x;', {baseStyle: 'Google', tabWidth: 2, useSpaces: false});
        expect(exec.execute).toHaveBeenCalledWith(
            '/usr/bin/clang-format',
            ['--style={BasedOnStyle: Google, IndentWidth: 2, UseTab: AlignWithSpaces}'],
            {input: 'int x;'},
        );
    });
});
