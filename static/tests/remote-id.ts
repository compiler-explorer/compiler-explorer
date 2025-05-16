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

/// <reference path="../../node_modules/cypress/types/cypress-global-vars.d.ts" />

import {getRemoteId} from '../../shared/remote-utils.js';

import {UrlTestCases} from '../../shared/url-testcases.js';

import {ITestable} from './frontend-testing.interfaces.js';

class RemoteIdTests implements ITestable {
    public readonly description: string = 'remoteId';

    public async run() {
        UrlTestCases.forEach(testCase => {
            if (getRemoteId(testCase.remoteUrl, testCase.language) !== testCase.expectedId) {
                throw new Error(
                    `Test case failed for language: ${testCase.language}, remoteUrl: ${testCase.remoteUrl}, expectedId: ${testCase.expectedId}`,
                );
            }
        });
    }
}

window.compilerExplorerFrontendTesting.add(new RemoteIdTests());
