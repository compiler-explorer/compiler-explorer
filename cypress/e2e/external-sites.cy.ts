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

import {serialiseState} from '../../shared/url-serialization.js';
import {assertNoConsoleOutput, stubConsoleOutput} from '../support/utils';

function visitEditor(lang: string, source: string) {
    const state = {
        version: 4,
        content: [
            {
                type: 'row',
                content: [
                    {
                        type: 'component',
                        componentName: 'codeEditor',
                        componentState: {id: 1, lang, source},
                        isClosable: true,
                    },
                ],
            },
        ],
    };
    cy.visit(`/#${serialiseState(state)}`, {
        onBeforeLoad: win => {
            stubConsoleOutput(win);
        },
    });
}

function externalSites() {
    return cy.get('.external-sites:visible');
}

describe('Open in external sites menu', () => {
    afterEach(() => {
        return cy.window().then(_win => {
            assertNoConsoleOutput();
        });
    });

    it('offers all three sites for C++ and fills in their links when opened', () => {
        visitEditor('c++', 'struct S {\n    char c;\n    int i;\n};\n');
        externalSites().find('.open-in-external').click();
        externalSites().find('.dropdown-menu').should('be.visible');
        externalSites()
            .find('.open-in-cppinsights')
            .should('be.visible')
            .and('have.attr', 'href')
            .and('include', 'code=');
        externalSites().find('.open-in-quickbench').should('be.visible').and('have.attr', 'href').and('include', '#');
        externalSites()
            .find('.open-in-abiexplorer')
            .should('be.visible')
            .and('have.attr', 'href')
            .and('match', /^https:\/\/abiexplorer\.org\/#./);
    });

    it('offers only ABI Explorer for C', () => {
        visitEditor('c', 'struct S {\n    char c;\n    long l;\n};\n');
        externalSites().find('.open-in-external').click();
        externalSites().find('.open-in-abiexplorer').should('be.visible');
        externalSites().find('.open-in-cppinsights').should('not.be.visible');
        externalSites().find('.open-in-quickbench').should('not.be.visible');
    });

    it('is hidden for languages none of the sites handle', () => {
        visitEditor('cmake', 'project(example)\n');
        cy.get('span.lm_title:visible').should('contain', 'CMake');
        cy.get('.vim-flag:visible').should('exist');
        cy.get('.external-sites:visible').should('not.exist');
    });
});
