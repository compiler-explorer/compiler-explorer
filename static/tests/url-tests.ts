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
import {CURRENT_LAYOUT_VERSION} from '../../static/components.interfaces.js';
import {deserialiseState, loadState} from '../../static/url.js';

// Helper functions for DRY testing
function createVersionedConfig(content: any[], additionalProps: any = {}) {
    return {
        version: CURRENT_LAYOUT_VERSION,
        content,
        ...additionalProps,
    };
}

function findComponentInConfig(config: any, componentName: string): any {
    const row = config.content[0];
    return row.content?.find((item: any) => item.type === 'component' && item.componentName === componentName);
}

function validateBasicConfig(config: any): void {
    expect(config).toBeDefined();
    expect(config.content).toBeDefined();
    expect(Array.isArray(config.content)).toBe(true);
}

function validateEditorPane(config: any, expectedSourceSnippets: string[]): any {
    const editorPane = findComponentInConfig(config, 'codeEditor');
    expect(editorPane).toBeDefined();

    for (const snippet of expectedSourceSnippets) {
        expect(editorPane.componentState.source).toContain(snippet);
    }

    return editorPane;
}

function validateCompilerPane(
    config: any,
    expectedCompiler: string,
    expectedOptions: string,
    expectedFilters?: Record<string, boolean>,
): any {
    const compilerPane = findComponentInConfig(config, 'compiler');
    expect(compilerPane).toBeDefined();
    expect(compilerPane.componentState.compiler).toBe(expectedCompiler);
    expect(compilerPane.componentState.options).toBe(expectedOptions);

    if (expectedFilters) {
        for (const [filterName, expectedValue] of Object.entries(expectedFilters)) {
            expect(compilerPane.componentState.filters[filterName]).toBe(expectedValue);
        }
    }

    return compilerPane;
}

describe('URL serialization/deserialization', () => {
    it('should decode a "Known good state" URL with many panes active', () => {
        // This is a real URL from cypress/e2e/frontend.cy.ts
        const url =
            'z:OYLghAFBqd5TB8IAsQGMD2ATApgUWwEsAXTAJwBoiQIAzIgG1wDsBDAW1xAHIBGHpTqYWJAMro2zEHwAsQkSQCqAZ1wAFAB68ADIIBWMyozYtQ6AKQAmAELWblNc3QkiI2q2wBhTIwCuHCwgVpSeADJELLgAcgEARrjkIPIADpgqpG4sPv6BwZRpGa4iEVGxHAlJ8k64LlliJGzkJDkBQSE1dSINTSSlMfGJyY6Nza15HaN9kQMVQ7IAlI6YfuTo3DwA9JsA1AAqAJ4puDsHK%2BQ7WHg7KIm4lDsUO4yYbNg7pju4mpwpzAB0Fh0AEFIiQdioAI5%2BJq4CBgnYsAILHYWADsdhBO2xO3IuBIqxYiICOwAVMSOBYAMyY4HogAiPCWjF4AFZBEEeHpKJheF57PYIed1qirFSBJQSLomUsANYgVk6Yy8WSCDgKpWc7m8niCFQgJVSrlMyhwWBoLAsYTkDimdbUDzEMjkIjYIwmMwASTdlls9mWq3WvG2%2ByOJzOq0uOBOtzxDyeLzeHyJ31%2BAKBoNEEOhsPhWaRHBR6NpONx%2BMJFLJFOptIZJpZPHZlC1gh1PitFFtLBFADUiLgAO6JHYQQikJ7WcULQRGvQLJa3N5DCDMlVq4Ks5vSnm8PUGyXSpZmqBoch%2BFQkFBEKg0CBYDgpJiJaKcDbAAW2HZ4OhsPyMEiCLgToUK6RjCKIEhSNwcgKKIqgaNoxqUIYfCOLgzjFEEECeOMQSoeEMzlJURiFJkIi4SR6RkSw/REUMqGdJhPRjL4bRGIx9RTLRgxJAxUwUXxvTcXMvFLCQeK4KBOiro2HLbjqABK57ggAEp6Ck7H2g7Du%2Bfqft%2Bv7/iOY7OqK4oPD4D5Phck58NOB7GvOlCLngSQrsqPCqpQ6pWJuLY7rqjj7rOMqUPKshKg2VJyUhOozoepqIMeqAYJgVnMDetD3o%2BGUgMABl/iQlB4AAbkQ6xaQOADyxz%2BQw/6JPqEBxNucSRE0By8BKbXsOQBxVXE%2Bi1EaEr3lwohVSwjCdUheBxH4wBeFIjD6vwgh4F2wDSLN17DUQJW4Kt3LfLUfgkBsEpguh26MEQcTkB1Ph4Nu4lEOqa1LHQJjAColU1awXWCOB4iSNIMHA/BWjbihximOYH4OLdcT6pASyYCkmGra2B3kC6eAo%2B5HHuNhLDeKxeT4aTwnEahpGYQJBRUZh1P0WhGGcb0DNEywzHTGUPHsfx5N4SMQmEQLdkBms0GUAOpgkDVJCeh2CUNk2/k6jsul2PpuA/oVxnATZYp2fFjkLrgS5uXKGoeV5Pl%2BfJu5BYaCXJWgF5sOgsrnmwwD3Le2XWS%2BXC8NrgoFf%2BgFG6BqEwyowobCGhzHKc5xRtcsb3I8FyJu8nypg%2B6YggiUIwniebggWRYYhmpZ4gS5BEgWVYFjWGZ1pKElSTJ6tOzwPTezsqh%2ByclU6QjX564Z4KjkbZmoTslk5cOtn2SFTkucuMn2xuW6xc7%2Bqu45iXIGgKwkCkZ0OneaUr%2BQIdvpPkcAaEMduqhwOQWD8gQ2oUNIRhgOB6KRAa9xitqXgVUzpX3BJgOgWtn7TwNhAZe1kF7r0PBbK2tAd7rl8vvSBgUj4OTnKfc0IAzwXltOgcgmBvgpBvkHDKj8w5IP1lHN%2B448ZgUUN/aCv9FCQ0QtyeO6E9rExwsLIwBF%2BYiUokULIDM6ZZBZrxNmEieZC1yCLbmvM1GC05tIwSzQDGS3ErgSSbppIeT7gfHgSkLw7AALJezoTsfAmgUimAyCITS/YhwXHDrrDhs8TIThNhZO%2B6C15mznNg1yuC7b4MdvYvcx94lhRAAANkirwaKhDWzOw3jbKkaJ/hijFGiVkABONENS%2BB8CsFYNEHkrDrj4DoTU/c4kmmSklAZZ8QAnXQGdCgN8mg/XUKYdCYgUCYAHJyUa0STCYWmVERgcyFn%2BWYUMfKyD/yUF2Q/V8ztjnQPEPMxZ24RnAnID9Z2IyGjEE5EDPhoMBGwWUP/ERBh3RwwwAjYwd0CZowxlkLGPIcZ40OvAJYCdAwywvC89ZsyrlLJnBJQGssQFgNsRAopPBPGnVMsEmwU9Qkjn5HpBwHj56xNIaFLe1ssmKmSd5PeGtD7BTdklNAwB2AkDmgtCxAcsorOfKcngZKKUz2jtw2OXz%2BEyEEXBH50N2LiK6FhKROiZFU3FvI2mTMlHGMZookohqaYaO1bzLmWqmJcStazT2LE9UmL5rMGmYlu7WPAYUgKABxaIwI9g7HpLgeawB9gSX8dpIJ7CZ6G24QvKJ6VV4m0webZyltEnuQbLvAhXLiE8pPu7EA14mESpOaHaVibCryudIq%2BOidgy7BTuGdOVwYx3HjLnV4%2BcUw/CLrgQEJcsxl1zAiauqJa5YhxA3CsLdyRtxpB3NEjIu6WNAh9fFAadRhDCD2ZxOx1JxsCYgmlsqUHhONuZJe1aMG9M3rm7e7KHb7u5RkvpfKQAoDYCoWU6FGA%2BDjIHatrC61Xpfo2kCH8lUfJVV84RGqGIOqyB4UmDNZFetZio8iZr8M0WdeovR2i2JofZt0J1cjrWupaGa%2BjZifXbr9Xu4tgaVJeAfXic9E9oMHLCfSyJD7013tNoyl9OD81rg5UWnpLtJM2zZVFAlAVn3kKQBQ9GRVwNicgzKmDXCm3wZbYittoZU4RguN2m4vac7PAHcmL4w6/ijozKXHMFdp3IlnSWBd5Ym6VhXQEduIJO6ip3QIf1xbFZ8YTQJylc8U22TTffJ9Smc3SeU3kzyKTP0lu/UeX9Aq2BCsjQtKt%2BmpWGcE7Bnhn93lQSQ3/BCqGbWYUw2Td1oQDW0bwyagjPWiNmI6xzN1FGxvUbFv10j5GKai1MSRmQLGrEGhi/3YNobw0VejePBLOtyUv2TaZVLon0sMpKVlvNeC5OpKIekyTmmUqMEYCVDgit1BEGOLdKIlUb48AALR0BYJgQHxAVCSHINgQHJUpB%2BFwID9gXAVA8G3Mcgz9bOEbThswQHKgDgVF8EddaAQUjA7/IwQHHAcB/guvVxV9VzrkEB5ES0ahoSsFcFIQH3iVBqFWhKL%2BiHwZCPVUhNAr33ufe%2B%2BhGY/2xEYSAgAMT8N2TrxCHVAWVngbQSoMhzSaAAdVdJeDchpfV4rVmpg9R6T1xa%2Bz9mY8XL2HevUZZLp2RNoIyhlq7zKkkFvy8Wx7G9ntoH/YB4DDRgBVfvpjxLcrjNwd4RBEXqrvltcAZqqjOqsNmpw3RdRRHlGDeI7NnPmi7WMfQ9NpbFePVcxo7h0SW61s2OtwVzj3GxB7EDS72rSXb2pvOzEzNz6ElvqD3dgroesGstywUkPmX5SslkP8Pg2St/b539v%2BQDZ2kcs6d0tJT3%2BlDJSIwwOHpgDA87GV7FQEFXwaZ4kQHzKScIea6LtVWfRFGEizdF3U72LXUDxBSDoXWH5yeAuVgVdwjkExOwiXvR9wzSnAn2uzfXlCsBqQ3133wK3wAA531OUFMSEw9y10A6BY89N48assdX4n8TMjBFdhoVc1dtVeBJ1vNRAFheA28e52N%2B4vBld%2B9B8k1PdkDF5UDxMs1MkA8ZM8sZ9l9yDeUhlSpyoxUIANCgxAomCU9Gs09v8M8UNs9KNNEutsM%2BsW8FFqJS8LVy8bDzDbV5tdFa8tEZsnDFsGMesmNltzFLdgC2QbdeAI0ypewAl%2BM3djtJDxM0sx90DMsFDbsP0VDS0yFy1I8gNXsvAOAOA49g56DE8G1k8Gsv8f5kNxd/9nDOsSZutJtC8JZzU7DCMy9RsyMjFfD3D9F/DvCm9PCi8VsBC2MQDNsuMl48iB8GCkC4jR9fdLt59kiSD5NT9VDs1sDCCN8rBGlCDZAykdi9i0QqQPIl8FMKDBkKEANdNxVqta1xCSj9CyizNpYLMO005IxbMs4%2B1HMkwC5XNi5MxwRuC4QfNCw/M64AtG5m4SQQtKQ11wsN1MVWMrdgiCtgQnF9t4CQkJDh8zsZC/dFjX0WVp91Q2U0iitw9KFlIOBK1aDCi7iGCGcX8msKjWsAFqips896iFtGijVmj6ZWiHD2jujXDK8XCBimj6N%2Bj68nDVtBDRj7FHFwRnEz1MT7iPdcTvdH0Fjs0liSTghsl/gdAxQqQGlCCmlCDsk18aliDySz8St0B0BsAycJlsBsA8R%2BdFYMM1pKBlovT3AfS4gXhvYVB/SMUjk0ouxsAAB9CAvWIgbQdyDHIo6IurYqfENgJgUM8FAMiUR4xnJ8ZHXAaMlQPwOgBgRMz6Fkz5Nk35agbxekMndrX7VgTAMM7FIspJB6aqHM8M5oRgRsh8bcRiICaZfnJJbmSwgvawwY41Bw%2Bw6iYU3PavLo5c5vWcvoxjdciWeFW6D7Xs7FT2EgbMjXCUUVQc0BJCQA/cPwV0dsn0kqdGE8706LIQ%2BxQNLwbjPYCSTYBSPYMIKY4ojU4TFA7U8fJIokwPWTHyKkf4OpLfHQc0wgtEPgKkWQRUEIO0q7eUPgcpFC7Y2QQgmpbJAioikik4kIwKbCnJKwf4NEbJeiqwCKKwRC5CtEffXgQ/dUY/WfTLA/SijAnGXxIIWQIAA%3D%3D';

        const config = deserialiseState(url);
        // Basic validation that we got a valid config, main thing we care about is that it doesn't throw an exception in validation.
        expect(config).toBeDefined();
        expect(config.content).toBeDefined();
        expect(Array.isArray(config.content)).toBe(true);
        expect(config.content?.length).toBeGreaterThan(0);
    });

    // Legacy URLs from etc/oldhash.txt
    it('should handle legacy version 2 URL with icc compiler', () => {
        // From git history: should be icc, "-O3 -std=c++0x", all filters but comments
        const url =
            '%7B%22version%22%3A2%2C%22source%22%3A%22%23include%20%3Cxmmintrin.h%3E%5Cn%5Cnvoid%20f(__m128%20a%2C%20__m128%20b)%5Cn%7B%5Cn%20%20%2F%2F%20I%20am%20a%20walrus.%5Cn%7D%22%2C%22compiler%22%3A%22%2Fhome%2Fmgodbolt%2Fapps%2Fintel-icc-oss%2Fbin%2Ficc%22%2C%22options%22%3A%22-O3%20-std%3Dc%2B%2B0x%22%2C%22filterAsm%22%3A%7B%22labels%22%3Atrue%2C%22directives%22%3Atrue%7D%7D';

        const config = deserialiseState(url);

        validateBasicConfig(config);
        validateEditorPane(config, ['#include <xmmintrin.h>', 'I am a walrus']);
        validateCompilerPane(config, '/home/mgodbolt/apps/intel-icc-oss/bin/icc', '-O3 -std=c++0x', {
            labels: true,
            directives: true,
        });
    });

    it('should handle legacy version 3 URL with GCC 7 and widget source', () => {
        // From git history: should be GCC 7, with widgets source. Binary mode off, all other labels on. -std=c++1z -O3
        const url =
            "compilers:!((compiler:g7snapshot,options:'-std%3Dc%2B%2B1z+-O3+',source:'%23include+%3Cvector%3E%0A%0Astruct+Widget+%7B%0A++int+n%3B%0A++double+x,+y%3B%0A++Widget(const+Widget%26+o)+:+x(o.x),+y(o.y),+n(o.n)+%7B%7D%0A++Widget(int+n,+double+x,+double+y)+:+n(n),+x(x),+y(y)+%7B%7D%0A%7D%3B%0A%0Astd::vector%3CWidget%3E+vector%3B%0Aconst+int+N+%3D+1002%3B%0Adouble+a+%3D+0.1%3B%0Adouble+b+%3D+0.2%3B%0A%0Avoid+demo()+%7B%0A++vector.reserve(N)%3B%0A++for+(int+i+%3D+01%3B+i+%3C+N%3B+%2B%2Bi)%0A++%7B%0A%09Widget+w+%7Bi,+a,+b%7D%3B%0A%09vector.push_back(w)%3B+//+or+vector.push_back(std::move(w))%0A++%7D%0A%7D%0A%0Aint+main()%0A%7B%0A+%0A+%0A%7D%0A')),filterAsm:(colouriseAsm:!t,commentOnly:!t,directives:!t,intel:!t,labels:!t),version:3";

        const config = deserialiseState(url);

        validateBasicConfig(config);
        validateEditorPane(config, ['struct Widget', 'vector.reserve(N)']);
        validateCompilerPane(config, 'g7snapshot', '-std=c++1z -O3 ', {
            commentOnly: true,
            directives: true,
            intel: true,
            labels: true,
        });
    });

    it('should handle legacy version 3 URL with base64 encoded source', () => {
        // From git history: should be 4.7.4, no options, binary, intel, colourise
        const url =
            "compilers:!((compiler:g474,options:'',sourcez:PQKgBALgpgzhYHsBmYDGCC2AHATrGAlggHYB0pamGUx8AFlHmEgjmADY0DmEdAXACgAhiNFjxEyVOkzZw2QsVLl85WvVrVG7TolbdB7fsMmlx0xennLNsddu37Dy0%2BenXbwx8%2B7vPo/7OfoGaIMACAgS0YBhCUQAUUfBCOFyoADSUxHBodClgICApXABuAJRgAN4CYGB4EACuOMRgAIwATADcAgC%2BQAA)),filterAsm:(binary:!t,colouriseAsm:!t,commentOnly:!t,directives:!t,intel:!t,labels:!t),version:3";

        const config = deserialiseState(url);

        validateBasicConfig(config);
        validateEditorPane(config, ['test of compression']);
        validateCompilerPane(config, 'g474', '', {
            binary: true,
            intel: true,
            commentOnly: true,
            directives: true,
            labels: true,
        });
    });

    describe('loadState validation', () => {
        describe('Input validation', () => {
            it('should throw for null input', () => {
                expect(() => loadState(null as any, false)).toThrow('Invalid state: must be an object');
            });

            it('should throw for undefined input', () => {
                expect(() => loadState(undefined as any, false)).toThrow('Invalid state: must be an object');
            });

            it('should throw for non-object input', () => {
                expect(() => loadState('string' as any, false)).toThrow('Invalid state: must be an object');
                expect(() => loadState(123 as any, false)).toThrow('Invalid state: must be an object');
            });

            it('should throw for objects without version property', () => {
                expect(() => loadState({content: []} as any, false)).toThrow(
                    'Invalid state: missing version information',
                );
            });

            it('should throw for array input', () => {
                expect(() => loadState([] as any, false)).toThrow('Invalid state: missing version information');
            });

            it('should throw for function input', () => {
                expect(() => loadState((() => {}) as any, false)).toThrow('Invalid state: must be an object');
            });

            it('should throw for boolean input', () => {
                expect(() => loadState(true as any, false)).toThrow('Invalid state: must be an object');
                expect(() => loadState(false as any, false)).toThrow('Invalid state: must be an object');
            });

            it('should throw for invalid version types', () => {
                expect(() => loadState({version: 'invalid', content: []} as any, false)).toThrow(
                    "Invalid version 'invalid'",
                );
                expect(() => loadState({version: [], content: []} as any, false)).toThrow("Invalid version ''");
                expect(() => loadState({version: {}, content: []} as any, false)).toThrow(
                    "Invalid version '[object Object]'",
                );
            });

            it('should throw for unsupported version numbers', () => {
                expect(() => loadState({version: 0, content: []} as any, false)).toThrow("Invalid version '0'");
                expect(() => loadState({version: 5, content: []} as any, false)).toThrow("Invalid version '5'");
                expect(() => loadState({version: -1, content: []} as any, false)).toThrow("Invalid version '-1'");
            });
        });

        describe('Valid configurations', () => {
            it('should accept configuration with empty content array', () => {
                const config = createVersionedConfig([]);
                const result = loadState(config, false);
                expect(result.content).toEqual([]);
            });

            it('should accept valid compiler component', () => {
                const config = createVersionedConfig([
                    {
                        type: 'component',
                        componentName: 'compiler',
                        componentState: {
                            source: 1,
                            lang: 'c++',
                        },
                    },
                ]);
                const result = loadState(config, false);
                expect(result.content).toEqual(config.content);
            });
        });

        // TODO(#7807): Add comprehensive validation tests covering:
        //
        // A. Extended Input Validation:
        //    - Array inputs (should throw)
        //    - Function inputs (should throw)
        //    - Invalid version types (string, array, etc.)
        //    - Unsupported version numbers
        //    - Version migration edge cases
        //
        // B. Content Array Validation:
        //    - Non-array content property
        //    - Content with mixed valid/invalid items
        //    - Empty vs populated content handling
        //
        // C. Item Structure Validation:
        //    - Items missing 'type' property
        //    - Items with non-string type property
        //    - Unknown item types beyond component/row/column/stack
        //
        // D. Component Validation:
        //    - Missing componentName property
        //    - Non-string componentName
        //    - Unknown component names (not in COMPONENT_NAMES list)
        //    - Missing componentState property
        //    - Non-object componentState
        //    - Component-specific state validation (future #7808)
        //
        // E. Layout Item Validation:
        //    - row/column/stack items missing content property
        //    - row/column/stack items with non-array content
        //    - Nested validation errors (deep nesting)
        //    - Mixed valid/invalid nested items
        //
        // F. Error Message Validation:
        //    - Verify exact error messages for each case
        //    - Test error message includes item index for nested errors
        //    - Test error propagation through nested structures
        //
        // G. Edge Cases:
        //    - Deeply nested valid structures
        //    - Large configuration objects
        //    - Boundary conditions (empty arrays, null values)
        //    - Mixed configuration types
    });
});
