// Copyright (c) 2017, Patrick Quist
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

const SymbolStore = require("./symbol-store").SymbolStore;
const Demangler = require("./demangler").Demangler;

class DemanglerPascal extends Demangler {
    constructor(demanglerExe, compiler) {
        super(demanglerExe, compiler);

        this.symbolStore = new SymbolStore();
        this.fixedsymbols = {};
        this.ignoredsymbols = [];

        this.initBasicSymbols();
    }

    initBasicSymbols() {
        this.fixedsymbols.OUTPUT_$$_init = 'unit_initialization';
        this.fixedsymbols.OUTPUT_$$_finalize = 'unit_finalization';
        this.fixedsymbols.OUTPUT_$$_init_implicit = 'unit_initialization_implicit';
        this.fixedsymbols.OUTPUT_$$_finalize_implicit = 'unit_finalization_implicit';
        this.fixedsymbols.OUTPUT_init = 'unit_initialization';
        this.fixedsymbols.OUTPUT_finalize = 'unit_finalization';
        this.fixedsymbols.OUTPUT_init_implicit = 'unit_initialization_implicit';
        this.fixedsymbols.OUTPUT_finalize_implicit = 'unit_finalization_implicit';

        this.ignoredsymbols = [
            ".L",
            "VMT_$", "INIT_$", "INIT$_$", "FINALIZE$_$", "RTTI_$",
            "VMT_OUTPUT_", "INIT$_OUTPUT", "RTTI_OUTPUT_", "FINALIZE$_OUTPUT",
            "_$",
            "DEBUGSTART_$", "DEBUGEND_$", "DBG_$", "DBG2_$", "DBGREF_$",
            "DEBUGSTART_OUTPUT", "DEBUGEND_OUTPUT", "DBG_OUTPUT_", "DBG2_OUTPUT_", "DBGREF_OUTPUT_"
        ];
    }

    shouldIgnoreSymbol(text) {
        for (let k in this.ignoredsymbols) {
            if (text.startsWith(this.ignoredsymbols[k])) {
                return true;
            }
        }

        return false;
    }

    composeReadableMethodSignature(unitname, classname, methodname, params) {
        let signature = "";

        if (classname !== "") signature = classname.toLowerCase() + ".";

        signature = signature + methodname.toLowerCase();
        signature = signature + "(" + params.toLowerCase() + ")";

        return signature;
    }

    demangle(text) {
        if (!text.endsWith(':')) return false;
        if (this.shouldIgnoreSymbol(text)) return false;

        text = text.substr(0, text.length - 1);

        for (const k in this.fixedsymbols) {
            if (text === k) {
                text = text.replace(k, this.fixedsymbols[k]);
                this.symbolStore.add(k, this.fixedsymbols[k]);
                return this.fixedsymbols[k];
            }
        }

        if (text.startsWith("U_$OUTPUT_$$_")) {
            const unmangledGlobalVar = text.substr(13).toLowerCase();
            this.symbolStore.add(text, unmangledGlobalVar);
            return unmangledGlobalVar;
        } else if (text.startsWith("U_OUTPUT_")) {
            const unmangledGlobalVar = text.substr(9).toLowerCase();
            this.symbolStore.add(text, unmangledGlobalVar);
            return unmangledGlobalVar;
        }

        let idx, paramtype = "", phase = 0;
        let unitname = "", classname = "", methodname = "", params = "", resulttype = "";

        idx = text.indexOf("$_$");
        if (idx !== -1) {
            unitname = text.substr(0, idx - 1);
            classname = text.substr(idx + 3, text.indexOf("_$_", idx + 2) - idx - 3);
        }

        let signature = "";
        idx = text.indexOf("_$$_");
        if (idx !== -1) {
            if (unitname === "") unitname = text.substr(0, idx - 1);
            signature = text.substr(idx + 3);
        }

        if (unitname === "") {
            idx = text.indexOf("OUTPUT_");
            if (idx !== -1) {
                unitname = "OUTPUT";

                idx = text.indexOf("_$__");
                if (idx !== -1) {
                    classname = text.substr(7, idx - 7);
                    signature = text.substr(idx + 3);
                } else {
                    signature = text.substr(6);
                }
            }
        }

        if (signature !== "") {
            for (idx = 1; idx < signature.length; idx++) {
                if (signature[idx] === '$') {
                    if (phase === 0) phase = 1;
                    else if (phase === 1) {
                        if (paramtype === "") phase = 2;
                        else if (params !== "") {
                            params = params + "," + paramtype;
                            paramtype = "";
                        } else if (params === "") {
                            params = paramtype;
                            paramtype = "";
                        }
                    }
                } else {
                    if (phase === 0) methodname = methodname + signature[idx];
                    else if (phase === 1) paramtype = paramtype + signature[idx];
                    else if (phase === 2) resulttype = resulttype + signature[idx];
                }
            }

            if (paramtype !== "") {
                if (params !== "") params = params + "," + paramtype;
                else params = paramtype;
            }
        }

        const unmangled = this.composeReadableMethodSignature(unitname, classname, methodname, params);
        this.symbolStore.add(text, unmangled);

        return unmangled;
    }

    addDemangleToCache(text) {
        this.demangle(text);
    }

    demangleIfNeeded(text) {
        if (text.includes('$')) {
            if (this.shouldIgnoreSymbol(text)) {
                return text;
            }

            const translations = this.symbolStore.listTranslations();
            for (const idx in translations) {
                text = text.replace(translations[idx][0], translations[idx][1]);
            }

            return text;
        } else {
            return text;
        }
    }

    async process(result, execOptions) {
        let options = execOptions || {};
        this.result = result;

        if (!this.symbolstore) {
            this.symbolstore = new SymbolStore();
            this.collectLabels();
        }

        options.input = this.getInput();

        return options.input;
    }
}

exports.Demangler = DemanglerPascal;
