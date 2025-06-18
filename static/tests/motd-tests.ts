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

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {isValidAd} from '../../static/motd.js';

describe('MOTD Tests', () => {
    describe('with fake timers', () => {
        beforeEach(() => {
            vi.useFakeTimers();
        });

        afterEach(() => {
            vi.useRealTimers();
        });

        it('should keep ad if now > from', () => {
            vi.setSystemTime(new Date('2022-01-08T00:00:00+00:00'));
            expect(isValidAd({filter: [], html: '', valid_from: '2022-01-01T00:00:00+00:00'}, 'langForTest')).toBe(
                true,
            );
        });

        it('should filter ad if now < from', () => {
            vi.setSystemTime(new Date('2022-01-08T00:00:00+00:00'));
            expect(isValidAd({filter: [], html: '', valid_from: '2022-01-16T00:00:00+00:00'}, 'langForTest')).toBe(
                false,
            );
        });

        it('should keep ad if now < until', () => {
            vi.setSystemTime(new Date('2022-01-08T00:00:00+00:00'));
            expect(isValidAd({filter: [], html: '', valid_until: '2022-01-16T00:00:00+00:00'}, 'langForTest')).toBe(
                true,
            );
        });

        it('should filter ad if now > until', () => {
            vi.setSystemTime(new Date('2022-01-20T00:00:00+00:00'));
            expect(isValidAd({filter: [], html: '', valid_until: '2022-01-16T00:00:00+00:00'}, 'langForTest')).toBe(
                false,
            );
        });

        it('should keep ad if from < now < until', () => {
            vi.setSystemTime(new Date('2022-01-08T00:00:00+00:00'));
            expect(
                isValidAd(
                    {
                        filter: [],
                        html: '',
                        valid_from: '2022-01-01T00:00:00+00:00',
                        valid_until: '2022-01-16T00:00:00+00:00',
                    },
                    'langForTest',
                ),
            ).toBe(true);
        });

        it('should filter ad if now < from < until', () => {
            vi.setSystemTime(new Date('2022-01-08T00:00:00+00:00'));
            expect(
                isValidAd(
                    {
                        filter: [],
                        html: '',
                        valid_from: '2022-01-10T00:00:00+00:00',
                        valid_until: '2022-01-16T00:00:00+00:00',
                    },
                    'langForTest',
                ),
            ).toBe(false);
        });

        it('should filter ad if from < until < now', () => {
            vi.setSystemTime(new Date('2022-01-20T00:00:00+00:00'));
            expect(
                isValidAd(
                    {
                        filter: [],
                        html: '',
                        valid_from: '2022-01-10T00:00:00+00:00',
                        valid_until: '2022-01-16T00:00:00+00:00',
                    },
                    'langForTest',
                ),
            ).toBe(false);
        });

        it('should filter ad if until < now < from', () => {
            vi.setSystemTime(new Date('2022-01-08T00:00:00+00:00'));
            expect(
                isValidAd(
                    {
                        filter: [],
                        html: '',
                        valid_from: '2022-01-16T00:00:00+00:00',
                        valid_until: '2022-01-10T00:00:00+00:00',
                    },
                    'langForTest',
                ),
            ).toBe(false);
        });

        it('should filter ad if from < now < until but filtered by lang', () => {
            vi.setSystemTime(new Date('2022-01-08T00:00:00+00:00'));
            expect(
                isValidAd(
                    {
                        filter: ['fakeLang'],
                        html: '',
                        valid_from: '2022-01-01T00:00:00+00:00',
                        valid_until: '2022-01-16T00:00:00+00:00',
                    },
                    'langForTest',
                ),
            ).toBe(false);
        });

        it('should keep ad if from < now < until and not filtered by lang', () => {
            vi.setSystemTime(new Date('2022-01-08T00:00:00+00:00'));
            expect(
                isValidAd(
                    {
                        filter: ['langForTest'],
                        html: '',
                        valid_from: '2022-01-01T00:00:00+00:00',
                        valid_until: '2022-01-16T00:00:00+00:00',
                    },
                    'langForTest',
                ),
            ).toBe(true);
        });

        it('should keep ad if from = now < until and not filtered by lang', () => {
            vi.setSystemTime(new Date('2022-01-08T00:00:00+00:00'));
            expect(
                isValidAd(
                    {
                        filter: ['langForTest'],
                        html: '',
                        valid_from: '2022-01-08T00:00:00+00:00',
                        valid_until: '2022-01-16T00:00:00+00:00',
                    },
                    'langForTest',
                ),
            ).toBe(true);
        });

        it('should keep ad if from < now = until and not filtered by lang', () => {
            vi.setSystemTime(new Date('2022-01-08T00:00:00+00:00'));
            expect(
                isValidAd(
                    {
                        filter: ['langForTest'],
                        html: '',
                        valid_from: '2022-01-01T00:00:00+00:00',
                        valid_until: '2022-01-18T00:00:00+00:00',
                    },
                    'langForTest',
                ),
            ).toBe(true);
        });

        it('should keep ad when now equals valid_until', () => {
            vi.setSystemTime(new Date('2022-01-16T00:00:00+00:00'));
            expect(
                isValidAd(
                    {
                        filter: [],
                        html: '',
                        valid_until: '2022-01-16T00:00:00+00:00',
                    },
                    'langForTest',
                ),
            ).toBe(true);
        });

        it('should keep ad when now equals valid_from', () => {
            vi.setSystemTime(new Date('2022-01-16T00:00:00+00:00'));
            expect(
                isValidAd(
                    {
                        filter: [],
                        html: '',
                        valid_from: '2022-01-16T00:00:00+00:00',
                    },
                    'langForTest',
                ),
            ).toBe(true);
        });
    });

    it('should keep ad if sublang is not set', () => {
        expect(isValidAd({filter: [], html: ''}, 'fakeLang')).toBe(true);
    });

    it('should keep ad if sublang is not set even if filtering for lang', () => {
        expect(isValidAd({filter: ['fakeLang'], html: ''}, 'langForTest')).toBe(false);
    });

    it('should keep ad if no lang is set', () => {
        expect(isValidAd({filter: [], html: ''}, 'langForTest')).toBe(true);
    });

    it('should filter ad if not the correct language', () => {
        expect(isValidAd({filter: ['anotherLang'], html: ''}, 'langForTest')).toBe(false);
    });

    it('should keep ad if the correct language is used', () => {
        expect(isValidAd({filter: ['langForTest'], html: ''}, 'langForTest')).toBe(true);
    });

    it('should keep ad if valid_from has invalid date format', () => {
        expect(
            isValidAd(
                {
                    filter: [],
                    html: '',
                    valid_from: 'invalid-date-format',
                },
                'langForTest',
            ),
        ).toBe(true);
    });

    it('should keep ad if valid_until has invalid date format', () => {
        expect(
            isValidAd(
                {
                    filter: [],
                    html: '',
                    valid_until: 'not-a-date',
                },
                'langForTest',
            ),
        ).toBe(true);
    });

    it('should keep ad if both dates have invalid format', () => {
        expect(
            isValidAd(
                {
                    filter: [],
                    html: '',
                    valid_from: 'invalid-from',
                    valid_until: 'invalid-until',
                },
                'langForTest',
            ),
        ).toBe(true);
    });

    it('should keep ad with empty string sublang', () => {
        expect(isValidAd({filter: [], html: ''}, '')).toBe(true);
    });

    it('should keep ad with null sublang', () => {
        expect(isValidAd({filter: [], html: ''}, null as any)).toBe(true);
    });

    it('should keep ad with undefined sublang', () => {
        expect(isValidAd({filter: [], html: ''}, undefined as any)).toBe(true);
    });

    it('should keep ad if sublang matches one of multiple filter languages', () => {
        expect(
            isValidAd(
                {
                    filter: ['lang1', 'lang2', 'langForTest', 'lang3'],
                    html: '',
                },
                'langForTest',
            ),
        ).toBe(true);
    });

    it('should filter ad if sublang does not match any of multiple filter languages', () => {
        expect(
            isValidAd(
                {
                    filter: ['lang1', 'lang2', 'lang3'],
                    html: '',
                },
                'langForTest',
            ),
        ).toBe(false);
    });

    it('should filter ad with invalid date and non-matching language filter', () => {
        expect(
            isValidAd(
                {
                    filter: ['anotherLang'],
                    html: '',
                    valid_from: 'invalid-date',
                },
                'langForTest',
            ),
        ).toBe(false);
    });

    it('should keep ad with invalid date and matching language filter', () => {
        expect(
            isValidAd(
                {
                    filter: ['langForTest'],
                    html: '',
                    valid_until: 'invalid-date',
                },
                'langForTest',
            ),
        ).toBe(true);
    });

    it('should handle filter with empty strings', () => {
        expect(
            isValidAd(
                {
                    filter: ['', 'langForTest', ''],
                    html: '',
                },
                'langForTest',
            ),
        ).toBe(true);
    });

    it('should handle filter with duplicate languages', () => {
        expect(
            isValidAd(
                {
                    filter: ['langForTest', 'langForTest', 'langForTest'],
                    html: '',
                },
                'langForTest',
            ),
        ).toBe(true);
    });
});
