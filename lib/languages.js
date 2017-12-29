// Copyright (c) 2012-2017, Matt Godbolt
//
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

/*  id is what we internally use to identify the language.
        *MUST BE* Same as the key so we can always query for it, everywhere.
            Used in: properties, url redirect, lang dropdown
    name is what we display to the user
            Used in: UI
    monaco is how we tell the monaco editor which language is this
            Used in: Monaco.editor.language
    extension is an array for the usual extensions of the language.
        The first one declared will be used as the default, else txt
        Leading point is needed
            Used in: Save to file extension
*/

const languages = {
    'c++': {
        id: 'c++',
        name: 'C++',
        monaco: 'cppp',
        extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c']
    },
    cppx: {
        id: 'cppx',
        name: 'Cppx',
        monaco: 'cppp',
        extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c']
    },
    /*c: {
        id: 'c',
        name: 'C',
        monaco: 'c',
        extensions: ['.c', '.h']
    },*/
    rust: {
        id: 'rust',
        name: 'Rust',
        monaco: 'rust',
        extensions: ['.rs']
    },
    d: {
        id: 'd',
        name: 'D',
        monaco: 'd',
        extensions: ['.d']
    },
    go: {
        id: 'go',
        name: 'Go',
        monaco: 'go',
        extensions: ['.go']
    },
    ispc: {
        id: 'ispc',
        name: 'ispc',
        monaco: 'ispc',
        extensions: ['.ispc']
    },
    haskell: {
        id: 'haskell',
        name: 'Haskell',
        monaco: 'haskell',
        extensions: ['.hs', '.haskell']
    },
    swift: {
        id: 'swift',
        name: 'Swift',
        monaco: 'swift',
        extensions: ['.swift']
    },
    pascal: {
        id: 'pascal',
        name: 'Pascal',
        monaco: 'pascal',
        extensions: ['.pas']
    },
    php: {
        id: 'php',
        name: 'PHP',
        monaco: 'php',
        extensions: ['.php']
    }
};

const fs = require('fs-extra');
const _ = require('underscore-node');
const path = require('path');
_.each(languages, lang => {
    try {
        const example = fs.readFileSync(path.join('examples', lang.id, 'default' + lang.extensions[0]), 'utf8');
        lang.example = example;
    } catch (error) {
        lang.example = "Oops, something went wrong and we could not get the default code for this language.";
    }
});

module.exports = {
    list: languages
};
