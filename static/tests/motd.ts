// Copyright (c) 2022, Compiler Explorer Authors
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

import {assert} from 'chai';
import {isValidAd} from '../motd.js';
import {ITestable} from './frontend-testing.interfaces.js';

import * as sinon from '../../node_modules/sinon/pkg/sinon-esm.js';

class MotdTests implements ITestable {
    public readonly description: string = 'motd';

    private static assertAd(ad, subLang, expected, message) {
        assert.equal(isValidAd(ad, subLang), expected, message);
    }

    private static assertAdWithDateNow(dateNow, ad, subLang, expected, message) {
        const dateNowStub = sinon.stub(Date, 'now');
        dateNowStub.returns(dateNow);
        MotdTests.assertAd(ad, subLang, expected, message);
        dateNowStub.restore();
    }

    public async run() {
        MotdTests.assertAd(
            {
                filter: [],
                html: '',
            },
            null,
            true,
            'Keep ad if sublang is not set',
        );

        MotdTests.assertAd(
            {
                filter: ['fakeLang'],
                html: '',
            },
            true,
            false,
            'Keep ad if sublang is not set even if filtering for lang',
        );

        MotdTests.assertAd(
            {
                filter: [],
                html: '',
            },
            'langForTest',
            true,
            'Keep ad if no lang is set',
        );

        MotdTests.assertAd(
            {
                filter: ['anotherLang'],
                html: '',
            },
            'langForTest',
            false,
            'Filters ad if not the correct language',
        );

        MotdTests.assertAd(
            {
                filter: ['langForTest'],
                html: '',
            },
            'langForTest',
            true,
            'Keep ad if the correct language is used',
        );

        MotdTests.assertAdWithDateNow(
            Date.parse('2022-01-08T00:00:00+00:00'),
            {
                filter: [],
                html: '',
                valid_from: '2022-01-01T00:00:00+00:00',
            },
            'langForTest',
            true,
            'Keep ad if now > from',
        );

        MotdTests.assertAdWithDateNow(
            Date.parse('2022-01-08T00:00:00+00:00'),
            {
                filter: [],
                html: '',
                valid_from: '2022-01-16T00:00:00+00:00',
            },
            'langForTest',
            false,
            'Filter ad if now < from',
        );

        MotdTests.assertAdWithDateNow(
            Date.parse('2022-01-08T00:00:00+00:00'),
            {
                filter: [],
                html: '',
                valid_until: '2022-01-16T00:00:00+00:00',
            },
            'langForTest',
            true,
            'Keep ad if now < until',
        );

        MotdTests.assertAdWithDateNow(
            Date.parse('2022-01-08T00:00:00+00:00'),
            {
                filter: [],
                html: '',
                valid_from: '2022-01-16T00:00:00+00:00',
            },
            'langForTest',
            false,
            'Filter ad if now > until',
        );

        MotdTests.assertAdWithDateNow(
            Date.parse('2022-01-08T00:00:00+00:00'),
            {
                filter: [],
                html: '',
                valid_from: '2022-01-01T00:00:00+00:00',
                valid_until: '2022-01-16T00:00:00+00:00',
            },
            'langForTest',
            true,
            'Keep ad if from < now < until',
        );

        MotdTests.assertAdWithDateNow(
            Date.parse('2022-01-08T00:00:00+00:00'),
            {
                filter: [],
                html: '',
                valid_from: '2022-01-10T00:00:00+00:00',
                valid_until: '2022-01-16T00:00:00+00:00',
            },
            'langForTest',
            false,
            'Filter ad if now < from < until',
        );

        MotdTests.assertAdWithDateNow(
            Date.parse('2022-01-20T00:00:00+00:00'),
            {
                filter: [],
                html: '',
                valid_from: '2022-01-10T00:00:00+00:00',
                valid_until: '2022-01-16T00:00:00+00:00',
            },
            'langForTest',
            false,
            'Filter ad if from < until < now',
        );

        MotdTests.assertAdWithDateNow(
            Date.parse('2022-01-08T00:00:00+00:00'),
            {
                filter: [],
                html: '',
                valid_from: '2022-01-16T00:00:00+00:00',
                valid_until: '2022-01-10T00:00:00+00:00',
            },
            'langForTest',
            false,
            'Filter ad if until < now < from',
        );

        MotdTests.assertAdWithDateNow(
            Date.parse('2022-01-08T00:00:00+00:00'),
            {
                filter: ['fakeLang'],
                html: '',
                valid_from: '2022-01-01T00:00:00+00:00',
                valid_until: '2022-01-16T00:00:00+00:00',
            },
            'langForTest',
            false,
            'Filter ad if from < now < until but filtered by lang',
        );

        MotdTests.assertAdWithDateNow(
            Date.parse('2022-01-08T00:00:00+00:00'),
            {
                filter: ['langForTest'],
                html: '',
                valid_from: '2022-01-01T00:00:00+00:00',
                valid_until: '2022-01-16T00:00:00+00:00',
            },
            'langForTest',
            true,
            'Keep ad if from < now < until and not filtered by lang',
        );

        MotdTests.assertAdWithDateNow(
            Date.parse('2022-01-08T00:00:00+00:00'),
            {
                filter: ['langForTest'],
                html: '',
                valid_from: '2022-01-08T00:00:00+00:00',
                valid_until: '2022-01-16T00:00:00+00:00',
            },
            'langForTest',
            true,
            'Keep ad if from = now < until and not filtered by lang',
        );

        MotdTests.assertAdWithDateNow(
            Date.parse('2022-01-08T00:00:00+00:00'),
            {
                filter: ['langForTest'],
                html: '',
                valid_from: '2022-01-01T00:00:00+00:00',
                valid_until: '2022-01-18T00:00:00+00:00',
            },
            'langForTest',
            true,
            'Keep ad if from < now = until and not filtered by lang',
        );
    }
}

window.compilerExplorerFrontendTesting.add(new MotdTests());
