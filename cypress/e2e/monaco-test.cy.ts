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

import {setMonacoEditorContent} from '../support/utils';

describe('Monaco Editor Utilities', () => {
    beforeEach(() => {
        cy.visit('/');
    });

    it('should set simple code content', () => {
        const simpleCode = 'int main() { return 42; }';
        setMonacoEditorContent(simpleCode);

        // Verify the content was set (basic check)
        cy.get('.monaco-editor').should('contain.text', 'main');
    });

    it('should set complex multi-line code content', () => {
        const complexCode = `#include <iostream>
#include <vector>

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5};
    
    for (int num : nums) {
        std::cout << num << std::endl;
    }
    
    return 0;
}`;
        setMonacoEditorContent(complexCode);

        // Verify the content was set (basic check)
        cy.get('.monaco-editor').should('contain.text', 'iostream');
        cy.get('.monaco-editor').should('contain.text', 'vector');
    });

    it('should handle special characters and quotes', () => {
        const codeWithSpecialChars = `const char* message = "Hello, \"World\"!";
int result = (x > 0) ? x : -x;`;
        setMonacoEditorContent(codeWithSpecialChars);

        // Verify the content was set (basic check)
        cy.get('.monaco-editor').should('contain.text', 'Hello');
    });
});
