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

// Extra information used to serialize the state
export interface GccDumpViewSelectedPass {
    // FIXME(dkm): this type needs to be refactored.
    // In particular, see in gccdump-view.ts:{constructor, getCurrentState}
    // There is a mix of 'selectedPass' being a filename_prefix and a
    // GccDumpViewSelectedPass object.
    filename_suffix: string | null;
    name: string | null;
    command_prefix: string | null;
    selectedPass: string | null;
}

// This should reflect the corresponding UI widget in gccdump.pug
// Each optionButton should have a matching boolean here.
export type GccDumpFiltersState = {
    treeDump: boolean;
    rtlDump: boolean;
    ipaDump: boolean;

    rawOption: boolean;
    slimOption: boolean;
    allOption: boolean;

    gimpleFeOption: boolean;
    addressOption: boolean;
    blocksOption: boolean;
    linenoOption: boolean;
    detailsOption: boolean;
    statsOption: boolean;
    uidOption: boolean;
    vopsOption: boolean;
};

// state = selected pass + all option flags
export type GccDumpViewState = GccDumpFiltersState & GccDumpViewSelectedPass;
