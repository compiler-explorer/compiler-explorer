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

import {describe, expect, it} from 'vitest';
import {deserialiseState} from '../url.js';

describe('Historical URL Backward Compatibility', () => {
    describe('Version 2 (JSON format from 2013)', () => {
        it('should deserialize ICC example with filters', () => {
            // From etc/oldhash.txt
            const urlHash =
                '%7B%22version%22%3A2%2C%22source%22%3A%22%23include%20%3Cxmmintrin.h%3E%5Cn%5Cnvoid%20f(__m128%20a%2C%20__m128%20b)%5Cn%7B%5Cn%20%20%2F%2F%20I%20am%20a%20walrus.%5Cn%7D%22%2C%22compiler%22%3A%22%2Fhome%2Fmgodbolt%2Fapps%2Fintel-icc-oss%2Fbin%2Ficc%22%2C%22options%22%3A%22-O3%20-std%3Dc%2B%2B0x%22%2C%22filterAsm%22%3A%7B%22labels%22%3Atrue%2C%22directives%22%3Atrue%7D%7D';

            const state = deserialiseState(urlHash);

            expect(state).toBeTruthy();
            expect(state.version).toBe(4);
            expect(state.content).toBeTruthy();
            expect(Array.isArray(state.content)).toBe(true);
            expect(state.content.length).toBeGreaterThan(0);

            const row = state.content[0];
            expect(row.type).toBe('row');
            expect(Array.isArray(row.content)).toBe(true);
            expect(row.content.length).toBe(2);

            const editor = row.content.find((c: any) => c.componentName === 'codeEditor');
            expect(editor).toBeTruthy();
            expect(editor.componentState.source).toContain('#include <xmmintrin.h>');
            expect(editor.componentState.source).toContain('I am a walrus');

            const compiler = row.content.find((c: any) => c.componentName === 'compiler');
            expect(compiler).toBeTruthy();
            expect(compiler.componentState.compiler).toBe('/home/mgodbolt/apps/intel-icc-oss/bin/icc');
            expect(compiler.componentState.options).toBe('-O3 -std=c++0x');

            const filters = compiler.componentState.filters;
            expect(filters.labels).toBe(true);
            expect(filters.directives).toBe(true);
        });
    });

    describe('Version 3 (Rison format)', () => {
        it('should deserialize GCC 7 widgets example', () => {
            // From etc/oldhash.txt
            const urlHash =
                "compilers:!((compiler:g7snapshot,options:'-std%3Dc%2B%2B1z+-O3+',source:'%23include+%3Cvector%3E%0A%0Astruct+Widget+%7B%0A++int+n%3B%0A++double+x,+y%3B%0A++Widget(const+Widget%26+o)+:+x(o.x),+y(o.y),+n(o.n)+%7B%7D%0A++Widget(int+n,+double+x,+double+y)+:+n(n),+x(x),+y(y)+%7B%7D%0A%7D%3B%0A%0Astd::vector%3CWidget%3E+vector%3B%0Aconst+int+N+%3D+1002%3B%0Adouble+a+%3D+0.1%3B%0Adouble+b+%3D+0.2%3B%0A%0Avoid+demo()+%7B%0A++vector.reserve(N)%3B%0A++for+(int+i+%3D+01%3B+i+%3C+N%3B+%2B%2Bi)%0A++%7B%0A%09Widget+w+%7Bi,+a,+b%7D%3B%0A%09vector.push_back(w)%3B+//+or+vector.push_back(std::move(w))%0A++%7D%0A%7D%0A%0Aint+main()%0A%7B%0A+%0A+%0A%7D%0A')),filterAsm:(colouriseAsm:!t,commentOnly:!t,directives:!t,intel:!t,labels:!t),version:3";

            const state = deserialiseState(urlHash);

            expect(state).toBeTruthy();
            expect(state.version).toBe(4);
            expect(state.content).toBeTruthy();
            expect(Array.isArray(state.content)).toBe(true);

            const row = state.content[0];
            expect(row.type).toBe('row');
            expect(Array.isArray(row.content)).toBe(true);
            expect(row.content.length).toBe(2);

            const editor = row.content.find((c: any) => c.componentName === 'codeEditor');
            expect(editor).toBeTruthy();
            expect(editor.componentState.source).toContain('#include <vector>');
            expect(editor.componentState.source).toContain('struct Widget');
            expect(editor.componentState.source).toContain('std::vector<Widget> vector');
            expect(editor.componentState.options.colouriseAsm).toBe(true);

            const compiler = row.content.find((c: any) => c.componentName === 'compiler');
            expect(compiler).toBeTruthy();
            expect(compiler.componentState.compiler).toBe('g7snapshot');
            expect(compiler.componentState.options).toBe('-std=c++1z -O3 ');

            const filters = compiler.componentState.filters;
            expect(filters.commentOnly).toBe(true);
            expect(filters.directives).toBe(true);
            expect(filters.intel).toBe(true);
            expect(filters.labels).toBe(true);
        });

        it('should deserialize GCC 4.7.4 with compressed source', () => {
            // From etc/oldhash.txt - compressed with lzstring
            const urlHash =
                'compilers:!((compiler:g474,options:%27%27,sourcez:PQKgBALgpgzhYHsBmYDGCC2AHATrGAlggHYB0pamGUx8AFlHmEgjmADY0DmEdAXACgAhiNFjxEyVOkzZw2QsVLl85WvVrVG7TolbdB7fsMmlx0xennLNsddu37Dy0%2BenXbwx8%2B7vPo/7OfoGaIMACAgS0YBhCUQAUUfBCOFyoADSUxHBodClgICApXABuAJRgAN4CYGB4EACuOMRgAIwATADcAgC%2BQAA)),filterAsm:(binary:!t,colouriseAsm:!t,commentOnly:!t,directives:!t,intel:!t,labels:!t),version:3';

            const state = deserialiseState(urlHash);

            expect(state).toBeTruthy();
            expect(state.version).toBe(4);
            expect(state.content).toBeTruthy();
            expect(Array.isArray(state.content)).toBe(true);

            const row = state.content[0];
            expect(row.type).toBe('row');
            expect(Array.isArray(row.content)).toBe(true);
            expect(row.content.length).toBe(2);

            const editor = row.content.find((c: any) => c.componentName === 'codeEditor');
            expect(editor).toBeTruthy();
            expect(editor.componentState.source).toBeTruthy();
            expect(typeof editor.componentState.source).toBe('string');
            expect(editor.componentState.source.length).toBeGreaterThan(0);
            expect(editor.componentState.source).toContain('int');
            expect(editor.componentState.options.colouriseAsm).toBe(true);

            const compiler = row.content.find((c: any) => c.componentName === 'compiler');
            expect(compiler).toBeTruthy();
            expect(compiler.componentState.compiler).toBe('g474');
            expect(compiler.componentState.options).toBe('');

            const filters = compiler.componentState.filters;
            expect(filters.binary).toBe(true);
            expect(filters.commentOnly).toBe(true);
            expect(filters.directives).toBe(true);
            expect(filters.intel).toBe(true);
            expect(filters.labels).toBe(true);
        });
    });

    describe('Invalid/Edge Cases', () => {
        it('should handle empty string', () => {
            const state = deserialiseState('');
            expect(state).toBe(false);
        });

        it('should handle invalid JSON', () => {
            expect(() => {
                deserialiseState('%7Binvalid');
            }).toThrow();
        });

        it('should handle invalid rison', () => {
            expect(() => {
                deserialiseState('invalid:rison:data');
            }).toThrow();
        });

        it('should handle corrupt lzstring data', () => {
            // State with 'z' field but invalid base64
            expect(() => {
                deserialiseState('(z:invalid_base64_data)');
            }).toThrow();
        });
    });
});
