// Copyright (c) 2018, Patrick Quist
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
"use strict";

const
    SymbolStore = require('./symbol-store').SymbolStore,
    _ = require('underscore-node'),
    utils = require('../lib/utils'),
    exec = require('../lib/exec'),
    logger = require('../lib/logger').logger,
    Demangler = require("./demangler").Demangler;

const LabelMetadata = [
    {'ident': 'C1', 'description': 'complete object constructor'},
    {'ident': 'C2', 'description': 'base object constructor'},
    {'ident': 'C3', 'description': 'complete object allocating constructor'},
    {'ident': 'D0', 'description': 'deleting destructor'},
    {'ident': 'D1', 'description': 'complete object destructor'},
    {'ident': 'D2', 'description': 'base object destructor'}
];

class DemanglerCPP extends Demangler {
    GetMetadata(symbol) {
        let metadata = [];

        for (let i = 0; i < LabelMetadata.length; ++i) {
            if (symbol.includes(LabelMetadata[i].ident)) {
                metadata.push(LabelMetadata[i]);
            }
        }
        
        return metadata;
    }
}

exports.DemanglerCPP = DemanglerCPP;
