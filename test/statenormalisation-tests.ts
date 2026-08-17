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

import fs from 'node:fs';

import {describe, expect, it} from 'vitest';

import {ClientState} from '../lib/clientstate.js';
import {ClientStateGoldenifier, ClientStateNormalizer} from '../lib/clientstate-normalizer.js';

describe('Normalizing clientstate', () => {
    it('Should translate 2 compilers GL layout to clientstate', () => {
        const normalizer = new ClientStateNormalizer();

        const data = JSON.parse(fs.readFileSync('test/state/twocompilers.json', {encoding: 'utf8'}));
        normalizer.fromGoldenLayout(data);

        const resultdata = JSON.parse(fs.readFileSync('test/state/twocompilers.normalized.json', {encoding: 'utf8'}));

        // note: this trick is to get rid of undefined parameters
        const normalized = JSON.parse(JSON.stringify(normalizer.normalized));

        expect(normalized).toEqual(resultdata);
    });

    it('Should recognize everything and kitchensink as well', () => {
        const normalizer = new ClientStateNormalizer();

        const data = JSON.parse(fs.readFileSync('test/state/andthekitchensink.json', {encoding: 'utf8'}));

        normalizer.fromGoldenLayout(data);

        const resultdata = JSON.parse(
            fs.readFileSync('test/state/andthekitchensink.normalized.json', {encoding: 'utf8'}),
        );

        const normalized = JSON.parse(JSON.stringify(normalizer.normalized));

        expect(normalized).toEqual(resultdata);
    });

    it('Should support conformanceview', () => {
        const normalizer = new ClientStateNormalizer();

        const data = JSON.parse(fs.readFileSync('test/state/conformanceview.json', {encoding: 'utf8'}));

        normalizer.fromGoldenLayout(data);

        const resultdata = JSON.parse(
            fs.readFileSync('test/state/conformanceview.normalized.json', {encoding: 'utf8'}),
        );

        const normalized = JSON.parse(JSON.stringify(normalizer.normalized));

        expect(normalized).toEqual(resultdata);
    });

    it('Should support executors', () => {
        const normalizer = new ClientStateNormalizer();

        const data = JSON.parse(fs.readFileSync('test/state/executor.json', {encoding: 'utf8'}));

        normalizer.fromGoldenLayout(data);

        const resultdata = JSON.parse(fs.readFileSync('test/state/executor.normalized.json', {encoding: 'utf8'}));

        const normalized = JSON.parse(JSON.stringify(normalizer.normalized));

        expect(normalized).toEqual(resultdata);
    });

    it('Should support newer features', () => {
        const normalizer = new ClientStateNormalizer();

        const data = JSON.parse(fs.readFileSync('test/state/executorwrap.json', {encoding: 'utf8'}));

        normalizer.fromGoldenLayout(data);

        const resultdata = JSON.parse(fs.readFileSync('test/state/executorwrap.normalized.json', {encoding: 'utf8'}));

        const normalized = JSON.parse(JSON.stringify(normalizer.normalized));

        expect(normalized).toEqual(resultdata);
    });

    it('Allow output without editor id', () => {
        const normalizer = new ClientStateNormalizer();
        const data = JSON.parse(fs.readFileSync('test/state/output-editor-id.json', {encoding: 'utf8'}));
        normalizer.fromGoldenLayout(data);

        const resultdata = JSON.parse(
            fs.readFileSync('test/state/output-editor-id.normalized.json', {encoding: 'utf8'}),
        );

        const normalized = JSON.parse(JSON.stringify(normalizer.normalized));

        expect(normalized).toEqual(resultdata);
    });
});

describe('ClientState parsing', () => {
    it('Should work without executors', () => {
        const state = new ClientState({
            sessions: [
                {id: 1, language: 'c++', source: 'int main() {}', compilers: [{id: 'g91', options: '-O3 -std=c++2a'}]},
            ],
        });

        expect(state.sessions[0].compilers.length).toEqual(1);
        expect(state.sessions[0].executors.length).toEqual(0);
    });

    it('Should work with executor', () => {
        const state = new ClientState({
            sessions: [
                {
                    id: 1,
                    language: 'c++',
                    source: 'int main() {}',
                    compilers: [],
                    executors: [
                        {
                            compiler: {id: 'g91', options: '-O3 -std=c++2a'},
                        },
                    ],
                },
            ],
        });

        expect(state.sessions[0].compilers.length).toEqual(0);
        expect(state.sessions[0].executors.length).toEqual(1);
    });

    it('Should not contain id-less compilers', () => {
        const jsonStr = fs.readFileSync('test/state/bug-2231.json', {encoding: 'utf8'});
        const state = new ClientState(JSON.parse(jsonStr));
        expect(state.sessions[0].compilers.length).toEqual(1);
    });
});

describe('Trees', () => {
    it('ClientState to GL', () => {
        const jsonStr = fs.readFileSync('test/state/tree.json', {encoding: 'utf8'});
        const state = new ClientState(JSON.parse(jsonStr));
        expect(state.trees.length).toEqual(1);

        const gl = new ClientStateGoldenifier();
        gl.fromClientState(state);

        const golden = JSON.parse(JSON.stringify(gl.golden));

        const resultdata = JSON.parse(fs.readFileSync('test/state/tree.goldenified.json', {encoding: 'utf8'}));
        expect(golden).toEqual(resultdata);
    });

    it('GL to ClientState', () => {
        const jsonStr = fs.readFileSync('test/state/tree-gl.json', {encoding: 'utf8'});
        const gl = JSON.parse(jsonStr);

        const normalizer = new ClientStateNormalizer();
        normalizer.fromGoldenLayout(gl);

        const normalized = JSON.parse(JSON.stringify(normalizer.normalized));

        const resultdata = JSON.parse(fs.readFileSync('test/state/tree.normalized.json', {encoding: 'utf8'}));

        expect(normalized).toEqual(resultdata);
    });

    it('GL to ClientState with correct output pane', () => {
        const jsonStr = fs.readFileSync('test/state/tree-gl-outputpane.json', {encoding: 'utf8'});
        const gl = JSON.parse(jsonStr);

        const normalizer = new ClientStateNormalizer();
        normalizer.fromGoldenLayout(gl);

        const normalized = JSON.parse(JSON.stringify(normalizer.normalized));

        const resultdata = JSON.parse(
            fs.readFileSync('test/state/tree-gl-outputpane.normalized.json', {encoding: 'utf8'}),
        );

        expect(normalized).toEqual(resultdata);
    });

    // A stack may only hold components; anything else renders as a blank, untitled tab.
    function expectNoNestedStacks(golden: any) {
        const check = (item: any) => {
            if (item.type === 'stack') {
                for (const child of item.content ?? []) {
                    expect(child.type).toEqual('component');
                }
            }
            for (const child of item.content ?? []) check(child);
        };
        check({content: golden.content});
    }

    function componentNamesOf(golden: any): string[] {
        const names: string[] = [];
        const collect = (item: any) => {
            if (item.componentName) names.push(item.componentName);
            for (const child of item.content ?? []) collect(child);
        };
        collect({content: golden.content});
        return names;
    }

    // https://godbolt.org/z/59q8GYa5z: a CMake tree whose only pane is an executor, with no
    // editor-attached compilers or executors either. The right hand pane came out empty.
    function goldenifyExecutorOnlyTree(mutate?: (state: ClientState) => void) {
        const jsonStr = fs.readFileSync('test/state/tree-executor-only.json', {encoding: 'utf8'});
        const state = new ClientState(JSON.parse(jsonStr));
        expect(state.trees.length).toEqual(1);
        mutate?.(state);

        const gl = new ClientStateGoldenifier();
        gl.fromClientState(state);

        const golden = JSON.parse(JSON.stringify(gl.golden));
        expectNoNestedStacks(golden);
        return golden;
    }

    it('ClientState to GL with only a tree executor', () => {
        const golden = goldenifyExecutorOnlyTree(state => {
            expect(state.trees[0].compilers.length).toEqual(0);
            expect(state.trees[0].executors.length).toEqual(1);
        });

        expect(componentNamesOf(golden)).toContain('executor');
    });

    it('ClientState to GL with only a tree compiler', () => {
        const treeJson = fs.readFileSync('test/state/tree.json', {encoding: 'utf8'});
        const treeCompiler = new ClientState(JSON.parse(treeJson)).trees[0].compilers[0];

        const golden = goldenifyExecutorOnlyTree(state => {
            state.trees[0].compilers = [treeCompiler];
            state.trees[0].executors = [];
        });

        const names = componentNamesOf(golden);
        expect(names).toContain('compiler');
        expect(names).not.toContain('executor');
    });

    it('ClientState to GL with neither tree compilers nor executors', () => {
        const golden = goldenifyExecutorOnlyTree(state => {
            state.trees[0].compilers = [];
            state.trees[0].executors = [];
        });

        // Nothing to show on the right, so no empty pane should be reserved for it.
        expect(componentNamesOf(golden)).toEqual(['tree', 'codeEditor', 'codeEditor', 'codeEditor']);
        expect(golden.content[0].content.length).toEqual(2);
    });

    it('ClientState to Mobile GL', () => {
        const jsonStr = fs.readFileSync('test/state/tree-mobile.json', {encoding: 'utf8'});
        const state = new ClientState(JSON.parse(jsonStr));
        expect(state.trees.length).toEqual(1);

        const gl = new ClientStateGoldenifier();
        const slides = gl.generatePresentationModeMobileViewerSlides(state);

        const golden = JSON.parse(JSON.stringify(slides));
        //fs.writeFileSync('test/state/tree-mobile.goldenified.json', JSON.stringify(golden));

        const resultdata = JSON.parse(fs.readFileSync('test/state/tree-mobile.goldenified.json', {encoding: 'utf8'}));
        expect(golden).toEqual(resultdata);
    });
});

describe('bug-6380', () => {
    it('Should goldenify properly', () => {
        const jsonStr = fs.readFileSync('test/state/bug-6380.json', {encoding: 'utf8'});
        const state = new ClientState(JSON.parse(jsonStr));

        const gl = new ClientStateGoldenifier();
        gl.fromClientState(state);

        const golden = JSON.parse(JSON.stringify(gl.golden));

        const normalizer = new ClientStateNormalizer();
        normalizer.fromGoldenLayout(golden);
    });
});

describe('overrides-and-runtimeTools', () => {
    it('Should normalize overrides and runtimetools', () => {
        const jsonGlStr = fs.readFileSync('test/state/libsegfault.json', {encoding: 'utf8'});
        const golden = JSON.parse(jsonGlStr);

        const normalizer = new ClientStateNormalizer();
        normalizer.fromGoldenLayout(golden);

        const normalized = JSON.parse(JSON.stringify(normalizer.normalized));

        const resultdata = JSON.parse(fs.readFileSync('test/state/libsegfault.normalized.json', {encoding: 'utf8'}));
        expect(normalized).toEqual(resultdata);
    });
});
