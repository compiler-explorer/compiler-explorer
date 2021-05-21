// Copyright (c) 2018, Compiler Explorer Authors
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

import { ClientState } from '../lib/clientstate';
import { ClientStateNormalizer } from '../lib/clientstate-normalizer';

import { fs } from './utils';

describe('Normalizing clientstate', () => {
    it('Should translate 2 compilers GL layout to clientstate', () => {
        const normalizer = new ClientStateNormalizer();

        const data =  JSON.parse(fs.readFileSync('test/state/twocompilers.json'));

        normalizer.fromGoldenLayout(data);

        const resultdata = JSON.parse(fs.readFileSync('test/state/twocompilers.json.normalized'));

        normalizer.normalized.should.deep.equal(resultdata);
    });

    it('Should recognize everything and kitchensink as well', () => {
        const normalizer = new ClientStateNormalizer();

        const data =  JSON.parse(fs.readFileSync('test/state/andthekitchensink.json'));

        normalizer.fromGoldenLayout(data);

        const resultdata = JSON.parse(fs.readFileSync('test/state/andthekitchensink.json.normalized'));

        normalizer.normalized.should.deep.equal(resultdata);
    });

    it('Should support conformanceview', () => {
        const normalizer = new ClientStateNormalizer();

        const data =  JSON.parse(fs.readFileSync('test/state/conformanceview.json'));

        normalizer.fromGoldenLayout(data);

        const resultdata = JSON.parse(fs.readFileSync('test/state/conformanceview.json.normalized'));

        normalizer.normalized.should.deep.equal(resultdata);
    });

    it('Should support executors', () => {
        const normalizer = new ClientStateNormalizer();

        const data =  JSON.parse(fs.readFileSync('test/state/executor.json'));

        normalizer.fromGoldenLayout(data);

        const resultdata = JSON.parse(fs.readFileSync('test/state/executor.json.normalized'));

        normalizer.normalized.should.deep.equal(resultdata);
    });

    it('Should support newer features', () => {
        const normalizer = new ClientStateNormalizer();

        const data =  JSON.parse(fs.readFileSync('test/state/executorwrap.json'));

        normalizer.fromGoldenLayout(data);

        const resultdata = JSON.parse(fs.readFileSync('test/state/executorwrap.json.normalized'));

        normalizer.normalized.should.deep.equal(resultdata);
    });
});

describe('ClientState parsing', () => {
    it('Should work without executors', () => {
        const state = new ClientState({
            sessions: [
                {id:1,
                    language:'c++',
                    source:'int main() {}',
                    compilers:[{id:'g91',options:'-O3 -std=c++2a'}],
                }],
        });

        state.sessions[0].compilers.length.should.equal(1);
        state.sessions[0].executors.length.should.equal(0);
    });

    it('Should work with executor', () => {
        const state = new ClientState({
            sessions: [
                {id:1,
                    language:'c++',
                    source:'int main() {}',
                    compilers: [],
                    executors:[{
                        compiler:{id:'g91',options:'-O3 -std=c++2a'},
                    }],
                }],
        });

        state.sessions[0].compilers.length.should.equal(0);
        state.sessions[0].executors.length.should.equal(1);
    });

    it('Should not contain id-less compilers', () => {
        const jsonStr = fs.readFileSync('test/state/bug-2231.json');
        const state = new ClientState(JSON.parse(jsonStr));
        state.sessions[0].compilers.length.should.equal(1);
    });
});
