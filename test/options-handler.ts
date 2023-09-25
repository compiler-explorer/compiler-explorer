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

import {fileURLToPath} from 'url';

import _ from 'underscore';

import {AppDefaultArguments} from '../app.js';
import {BaseCompiler} from '../lib/base-compiler.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {ClientOptionsHandler, ClientOptionsType} from '../lib/options-handler.js';
import * as properties from '../lib/properties.js';
import {BaseTool} from '../lib/tooling/base-tool.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {LanguageKey} from '../types/languages.interfaces.js';

import {makeFakeCompilerInfo, should} from './utils.js';

const languages = {
    fake: {
        id: 'fake',
    },
};

// For tooling, we need to at least be able to stat a valid file.
// This is the one we know exists
const CURRENT_FILE_PATH = fileURLToPath(import.meta.url);

const optionsProps = {
    libs: 'fakelib:fs:someotherlib',
    'libs.fakelib.name': 'fake lib',
    'libs.fakelib.description': 'Its is a real, fake lib!',
    'libs.fakelib.versions': 'onePath:twoPaths:noPaths',
    'libs.fakelib.url': 'https://godbolt.org',
    'libs.fakelib.examples': 'abc:def',
    'libs.fakelib.versions.onePath.version': 'one path',
    'libs.fakelib.versions.onePath.path': '/dev/null',
    'libs.fakelib.versions.onePath.libpath': '/lib/null',
    'libs.fakelib.versions.onePath.liblink': 'hello',
    'libs.fakelib.versions.twoPaths.version': 'two paths',
    'libs.fakelib.versions.twoPaths.path': '/dev/null:/dev/urandom',
    'libs.fakelib.versions.twoPaths.libpath': '/lib/null:/lib/urandom',
    'libs.fakelib.versions.twoPaths.liblink': 'hello1:hello2',
    'libs.fakelib.versions.noPaths.version': 'no paths',
    'libs.fakelib.versions.noPaths.path': '',
    'libs.fakelib.versions.noPaths.lookupversion': 'no-paths123',
    'libs.fakelib.versions.noPaths.options': '-DHELLO123 -DETC "--some thing with spaces"',
    'libs.fs.versions': 'std',
    'libs.fs.versions.std.version': 'std',
    'libs.fs.versions.std.staticliblink': 'c++fs:rt',
    'libs.fs.versions.std.dependencies': 'pthread',
    'libs.someotherlib.versions': 'trunk',
    'libs.someotherlib.versions.trunk.version': 'trunk',
    'libs.someotherlib.versions.trunk.staticliblink': 'someotherlib',
    'libs.someotherlib.versions.trunk.dependencies': 'c++fs',
    'libs.someotherlib.versions.trunk.alias': 'master',
    'libs.someotherlib.versions.trunk.hidden': true,
    tools: 'faketool:someothertool',
    'tools.faketool.name': 'Fake Tool',
    'tools.faketool.exe': CURRENT_FILE_PATH,
    'tools.faketool.type': 'independent',
    'tools.faketool.class': 'testing-tool',
    'tools.faketool.stdinHint': 'disabled',
    'tools.someothertool.name': 'Some Other Tool',
    'tools.someothertool.exe': CURRENT_FILE_PATH,
    'tools.someothertool.type': 'independent',
    'tools.someothertool.class': 'testing-tool',
    'tools.someothertool.stdinHint': 'disabled',
};

if (process.platform === 'win32') {
    optionsProps['libs.fakelib.versions.twoPaths.path'] = optionsProps['libs.fakelib.versions.twoPaths.path'].replace(
        ':',
        ';',
    );
    optionsProps['libs.fakelib.versions.twoPaths.libpath'] = optionsProps[
        'libs.fakelib.versions.twoPaths.libpath'
    ].replace(':', ';');
}

const moreLibProps = {
    libs: 'fs:someotherlib:yalib:fourthlib:autolib',

    'libs.fs.versions': 'std',
    'libs.fs.versions.std.version': 'std',
    'libs.fs.versions.std.staticliblink': 'fsextra:c++fs:rt',
    'libs.fs.versions.std.dependencies': 'pthread',

    'libs.someotherlib.versions': 'trunk',
    'libs.someotherlib.versions.trunk.version': 'trunk',
    'libs.someotherlib.versions.trunk.staticliblink': 'someotherlib',
    'libs.someotherlib.versions.trunk.dependencies': 'c++fs',
    'libs.someotherlib.versions.trunk.alias': 'master',

    'libs.yalib.versions': 'trunk',
    'libs.yalib.versions.trunk.version': 'trunk',
    'libs.yalib.versions.trunk.staticliblink': 'yalib',
    'libs.yalib.versions.trunk.dependencies': 'someotherlib:c++fs',

    'libs.fourthlib.versions': 'trunk',
    'libs.fourthlib.versions.trunk.version': 'trunk',
    'libs.fourthlib.versions.trunk.staticliblink': 'fourthlib:yalib:rt',

    'libs.autolib.versions': 'autodetect',
    'libs.autolib.versions.autodetect.version': 'autodetect',
    'libs.autolib.versions.autodetect.staticliblink': 'hello',
};

const fakeCompilerInfo = (id: string, lang: string, group: string, semver: string, isSemver: boolean) => {
    return makeFakeCompilerInfo({
        id: id,
        exe: '/dev/null',
        name: id,
        lang: lang as LanguageKey,
        group: group,
        isSemVer: isSemver,
        semver: semver,
        libsArr: [],
    });
};

class TestBaseCompiler extends BaseCompiler {
    public override supportedLibraries = super.supportedLibraries;
}

describe('Options handler', () => {
    let fakeOptionProps: ReturnType<typeof properties.fakeProps>;
    let compilerProps: properties.CompilerProps;
    let optionsHandler: ClientOptionsHandler;

    let fakeMoreCompilerProps: ReturnType<typeof properties.fakeProps>;
    let moreCompilerProps: properties.CompilerProps;
    let moreOptionsHandler: ClientOptionsHandler;

    let env: CompilationEnvironment;

    function createClientOptions(libs: ReturnType<ClientOptionsHandler['parseLibraries']>) {
        return {
            libs: {
                'c++': libs.fake,
            },
        } as unknown as ClientOptionsType;
    }

    before(() => {
        fakeOptionProps = properties.fakeProps(optionsProps);
        compilerProps = new properties.CompilerProps(languages, fakeOptionProps);
        optionsHandler = new ClientOptionsHandler([], compilerProps, {env: ['dev']} as unknown as AppDefaultArguments);

        fakeMoreCompilerProps = properties.fakeProps(moreLibProps);
        moreCompilerProps = new properties.CompilerProps(languages, fakeMoreCompilerProps);
        moreOptionsHandler = new ClientOptionsHandler([], moreCompilerProps, {
            env: ['dev'],
        } as unknown as AppDefaultArguments);

        env = {
            ceProps: properties.fakeProps({}),
            compilerProps: () => {},
        } as unknown as CompilationEnvironment;
    });

    it('should always return an array of paths', () => {
        const libs = optionsHandler.parseLibraries({fake: optionsProps.libs});
        _.each(libs[languages.fake.id]['fakelib'].versions, version => {
            Array.isArray(version.path).should.equal(true);
        });
        libs.should.deep.equal({
            fake: {
                fakelib: {
                    description: 'Its is a real, fake lib!',
                    name: 'fake lib',
                    url: 'https://godbolt.org',
                    dependencies: [],
                    liblink: [],
                    staticliblink: [],
                    examples: ['abc', 'def'],
                    options: [],
                    packagedheaders: false,
                    versions: {
                        noPaths: {
                            $order: 2,
                            path: [],
                            version: 'no paths',
                            liblink: [],
                            libpath: [],
                            staticliblink: [],
                            dependencies: [],
                            alias: [],
                            lookupversion: 'no-paths123',
                            options: ['-DHELLO123', '-DETC', '--some thing with spaces'],
                            hidden: false,
                            packagedheaders: false,
                        },
                        onePath: {
                            $order: 0,
                            path: ['/dev/null'],
                            version: 'one path',
                            staticliblink: [],
                            dependencies: [],
                            liblink: ['hello'],
                            libpath: ['/lib/null'],
                            alias: [],
                            options: [],
                            hidden: false,
                            packagedheaders: false,
                        },
                        twoPaths: {
                            $order: 1,
                            path: ['/dev/null', '/dev/urandom'],
                            staticliblink: [],
                            dependencies: [],
                            liblink: ['hello1', 'hello2'],
                            libpath: ['/lib/null', '/lib/urandom'],
                            version: 'two paths',
                            alias: [],
                            options: [],
                            hidden: false,
                            packagedheaders: false,
                        },
                    },
                },
                fs: {
                    description: undefined,
                    name: undefined,
                    url: undefined,
                    dependencies: [],
                    liblink: [],
                    staticliblink: [],
                    examples: [],
                    options: [],
                    packagedheaders: false,
                    versions: {
                        std: {
                            $order: 0,
                            libpath: [],
                            path: [],
                            version: 'std',
                            alias: [],
                            liblink: [],
                            staticliblink: ['c++fs', 'rt'],
                            dependencies: ['pthread'],
                            options: [],
                            hidden: false,
                            packagedheaders: false,
                        },
                    },
                },
                someotherlib: {
                    description: undefined,
                    name: undefined,
                    url: undefined,
                    dependencies: [],
                    liblink: [],
                    staticliblink: [],
                    examples: [],
                    options: [],
                    packagedheaders: false,
                    versions: {
                        trunk: {
                            $order: 0,
                            libpath: [],
                            path: [],
                            version: 'trunk',
                            alias: ['master'],
                            liblink: [],
                            staticliblink: ['someotherlib'],
                            dependencies: ['c++fs'],
                            options: [],
                            hidden: true,
                            packagedheaders: false,
                        },
                    },
                },
            },
        });
    });
    it('should order compilers as expected', () => {
        const compilers = [
            fakeCompilerInfo('a1', languages.fake.id, 'a', '0.0.1', true),
            fakeCompilerInfo('a2', languages.fake.id, 'a', '0.2.0', true),
            fakeCompilerInfo('a3', languages.fake.id, 'a', '0.2.1', true),

            fakeCompilerInfo('b1', languages.fake.id, 'b', 'trunk', true),
            fakeCompilerInfo('b2', languages.fake.id, 'b', '1.0.0', true),
            fakeCompilerInfo('b3', languages.fake.id, 'b', '0.5.0', true),

            fakeCompilerInfo('c1', languages.fake.id, 'c', '3.0.0', true),
            fakeCompilerInfo('c2', languages.fake.id, 'c', '3.0.0', true),
            fakeCompilerInfo('c3', languages.fake.id, 'c', '3.0.0', true),

            fakeCompilerInfo('d1', languages.fake.id, 'd', '1', true),
            fakeCompilerInfo('d2', languages.fake.id, 'd', '2.0.0', true),
            fakeCompilerInfo('d3', languages.fake.id, 'd', '0.0.5', true),

            fakeCompilerInfo('e1', languages.fake.id, 'e', '0..0', false),
            fakeCompilerInfo('e2', languages.fake.id, 'e', '', false),

            fakeCompilerInfo('f1', languages.fake.id, 'f', '5', true),
            fakeCompilerInfo('f2', languages.fake.id, 'f', '5.1', true),
            fakeCompilerInfo('f3', languages.fake.id, 'f', '5.2', true),

            fakeCompilerInfo('g1', languages.fake.id, 'g', '5 a', true),
            fakeCompilerInfo('g2', languages.fake.id, 'g', '5.1 b d', true),
            fakeCompilerInfo('g3', languages.fake.id, 'g', '5.2 ce fg', true),
        ];
        const expectedOrder = {
            a: {
                a1: -0,
                a2: -1,
                a3: -2,
            },
            b: {
                b1: -2,
                b2: -1,
                b3: -0,
            },
            c: {
                c1: -0,
                c2: -1,
                c3: -2,
            },
            d: {
                d1: -1,
                d2: -2,
                d3: -0,
            },
            e: {
                e1: undefined,
                e2: undefined,
            },
            f: {
                f1: -0,
                f2: -1,
                f3: -2,
            },
            g: {
                g1: -0,
                g2: -1,
                g3: -2,
            },
        };
        optionsHandler.setCompilers(compilers);
        _.each(optionsHandler.get().compilers, compiler => {
            should.equal(
                compiler['$order'],
                expectedOrder[(compiler as CompilerInfo).group][(compiler as CompilerInfo).id],
                `group: ${(compiler as CompilerInfo).group} id: ${(compiler as CompilerInfo).id}`,
            );
        });
        optionsHandler.setCompilers([]);
    });
    it('should get static libraries', () => {
        const libs = optionsHandler.parseLibraries({fake: optionsProps.libs});
        const compilerInfo = fakeCompilerInfo('g82', 'c++', 'cpp', '8.2', true);

        const clientOptions = createClientOptions(libs);

        const compiler = new BaseCompiler(compilerInfo, env);
        compiler.initialiseLibraries(clientOptions);

        const staticlinks = compiler.getStaticLibraryLinks([{id: 'fs', version: 'std'}]);
        staticlinks.should.deep.equal(['-lc++fs', '-lrt', '-lpthread']);

        const sharedlinks = compiler.getSharedLibraryLinks([{id: 'fs', version: 'std'}]);
        sharedlinks.should.deep.equal([]);
    });
    it('should sort static libraries', () => {
        const libs = optionsHandler.parseLibraries({fake: optionsProps.libs});
        const compilerInfo = fakeCompilerInfo('g82', 'c++', 'cpp', '8.2', true);

        const clientOptions = createClientOptions(libs);

        const compiler = new BaseCompiler(compilerInfo, env);
        compiler.initialiseLibraries(clientOptions);

        let staticlinks = compiler.getSortedStaticLibraries([{id: 'someotherlib', version: 'trunk'}]);
        staticlinks.should.deep.equal(['someotherlib', 'c++fs']);

        staticlinks = compiler.getSortedStaticLibraries([
            {id: 'fs', version: 'std'},
            {id: 'someotherlib', version: 'trunk'},
        ]);
        staticlinks.should.deep.equal(['someotherlib', 'c++fs', 'rt', 'pthread']);
    });
    it('library sort special case 1', () => {
        const libs = moreOptionsHandler.parseLibraries({fake: moreLibProps.libs});
        const compilerInfo = fakeCompilerInfo('g82', 'c++', 'cpp', '8.2', true);

        const compiler = new BaseCompiler(compilerInfo, env);

        const clientOptions = createClientOptions(libs);
        compiler.initialiseLibraries(clientOptions);

        const staticlinks = compiler.getSortedStaticLibraries([{id: 'fs', version: 'std'}]);
        staticlinks.should.deep.equal(['fsextra', 'c++fs', 'rt', 'pthread']);
    });
    it('library sort special case 2', () => {
        const libs = moreOptionsHandler.parseLibraries({fake: moreLibProps.libs});
        const compilerInfo = fakeCompilerInfo('g82', 'c++', 'cpp', '8.2', true);

        const compiler = new BaseCompiler(compilerInfo, env);

        const clientOptions = createClientOptions(libs);
        compiler.initialiseLibraries(clientOptions);

        const staticlinks = compiler.getSortedStaticLibraries([
            {id: 'yalib', version: 'trunk'},
            {id: 'fs', version: 'std'},
            {id: 'someotherlib', version: 'trunk'},
        ]);
        staticlinks.should.deep.equal(['yalib', 'someotherlib', 'fsextra', 'c++fs', 'rt', 'pthread']);
    });
    it('library sort special case 3', () => {
        const libs = moreOptionsHandler.parseLibraries({fake: moreLibProps.libs});
        const compilerInfo = fakeCompilerInfo('g82', 'c++', 'cpp', '8.2', true);

        const compiler = new BaseCompiler(compilerInfo, env);

        const clientOptions = createClientOptions(libs);
        compiler.initialiseLibraries(clientOptions);

        const staticlinks = compiler.getSortedStaticLibraries([
            {id: 'fourthlib', version: 'trunk'},
            {id: 'fs', version: 'std'},
            {id: 'someotherlib', version: 'trunk'},
        ]);
        staticlinks.should.deep.equal(['fourthlib', 'yalib', 'someotherlib', 'fsextra', 'c++fs', 'rt', 'pthread']);
    });
    it('filtered library list', () => {
        const libs = moreOptionsHandler.parseLibraries({fake: moreLibProps.libs});
        const compilerInfo = fakeCompilerInfo('g82', 'c++', 'cpp', '8.2', true);

        compilerInfo.libsArr = ['fs.std', 'someotherlib'];

        const compiler = new TestBaseCompiler(compilerInfo, env);

        const clientOptions = createClientOptions(libs);
        compiler.initialiseLibraries(clientOptions);

        const libNames = _.keys(compiler.supportedLibraries);
        libNames.should.deep.equal(['fs', 'someotherlib']);
    });
    it('can detect libraries from options', () => {
        const libs = moreOptionsHandler.parseLibraries({fake: moreLibProps.libs});
        const compilerInfo = fakeCompilerInfo('g82', 'c++', 'cpp', '8.2', true);

        const compiler = new BaseCompiler(compilerInfo, env);

        const clientOptions = createClientOptions(libs);
        compiler.initialiseLibraries(clientOptions);

        const obj = {
            libraries: [{id: 'ctre', version: 'trunk'}],
            options: ['-O3', '--std=c++17', '-lhello'],
        };
        compiler.tryAutodetectLibraries(obj).should.equal(true);

        obj.libraries.should.deep.equal([
            {id: 'ctre', version: 'trunk'},
            {id: 'autolib', version: 'autodetect'},
        ]);
        obj.options.should.deep.equal(['-O3', '--std=c++17']);
    });
    it("server-side library alias support (just in case client doesn't support it)", () => {
        const libs = moreOptionsHandler.parseLibraries({fake: moreLibProps.libs});
        const compilerInfo = fakeCompilerInfo('g82', 'c++', 'cpp', '8.2', true);
        const env = {
            ceProps: properties.fakeProps({}),
            compilerProps: () => {},
        } as unknown as CompilationEnvironment;

        const compiler = new BaseCompiler(compilerInfo, env);

        const clientOptions = createClientOptions(libs);
        compiler.initialiseLibraries(clientOptions);

        const staticlinks = compiler.getSortedStaticLibraries([{id: 'someotherlib', version: 'master'}]);
        staticlinks.should.deep.equal(['someotherlib', 'c++fs']);
    });
    it('should be able to parse basic tools', () => {
        class TestBaseTool extends BaseTool {
            public override env = super.env;
        }
        const tools = optionsHandler.parseTools({fake: optionsProps.tools}) as unknown as Record<
            string,
            Record<string, Partial<TestBaseTool>>
        >;

        _.each(tools.fake, tool => {
            delete tool.env;
        });

        tools.should.deep.equal({
            fake: {
                faketool: {
                    id: 'faketool',
                    type: 'independent',
                    addOptionsToToolArgs: true,
                    tool: {
                        args: undefined,
                        compilerLanguage: 'fake',
                        exclude: [],
                        exe: CURRENT_FILE_PATH,
                        id: 'faketool',
                        includeKey: undefined,
                        languageId: undefined,
                        monacoStdin: undefined,
                        name: 'Fake Tool',
                        options: [],
                        stdinHint: 'disabled',
                        type: 'independent',
                        icon: undefined,
                        darkIcon: undefined,
                    },
                },
                someothertool: {
                    id: 'someothertool',
                    type: 'independent',
                    addOptionsToToolArgs: true,
                    tool: {
                        args: undefined,
                        compilerLanguage: 'fake',
                        exclude: [],
                        exe: CURRENT_FILE_PATH,
                        id: 'someothertool',
                        includeKey: undefined,
                        languageId: undefined,
                        monacoStdin: undefined,
                        name: 'Some Other Tool',
                        options: [],
                        stdinHint: 'disabled',
                        type: 'independent',
                        icon: undefined,
                        darkIcon: undefined,
                    },
                },
            },
        });
    });
});
