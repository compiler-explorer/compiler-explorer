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

export class ClientStateCompilerOptions {
    binary = false;
    commentOnly = true;
    demangle = true;
    directives = true;
    execute = false;
    intel = true;
    labels = true;
    trim = false;
    debugCalls = false;

    constructor(jsondata?) {
        if (jsondata) this.fromJsonData(jsondata);
    }

    fromJsonData(jsondata) {
        if (jsondata.binary !== undefined) this.binary = jsondata.binary;
        if (jsondata.commentOnly !== undefined) this.commentOnly = jsondata.commentOnly;
        if (jsondata.demangle !== undefined) this.demangle = jsondata.demangle;
        if (jsondata.directives !== undefined) this.directives = jsondata.directives;
        if (jsondata.execute !== undefined) this.execute = jsondata.execute;
        if (jsondata.intel !== undefined) this.intel = jsondata.intel;
        if (jsondata.labels !== undefined) this.labels = jsondata.labels;
        if (jsondata.trim !== undefined) this.trim = jsondata.trim;
        if (jsondata.debugCalls !== undefined) this.debugCalls = jsondata.debugCalls;
    }
}

export class ClientStateCompiler {
    _internalid: any = undefined;
    id = '';
    options = '';
    filters: ClientStateCompilerOptions;
    libs: any[] = [];
    specialoutputs: any[] = [];
    tools: any[] = [];

    constructor(jsondata?) {
        if (jsondata) {
            this.filters = undefined as any as ClientStateCompilerOptions;
            this.fromJsonData(jsondata);
        } else {
            this.filters = new ClientStateCompilerOptions();
        }
    }

    fromJsonData(jsondata) {
        if (jsondata._internalid !== undefined) {
            this._internalid = jsondata._internalid;
        }

        if (jsondata.id !== undefined) {
            this.id = jsondata.id;
        } else if (jsondata.compilerId === undefined) {
            this.id = '';
        } else {
            this.id = jsondata.compilerId;
        }

        this.options = jsondata.options;
        this.filters = new ClientStateCompilerOptions(jsondata.filters);
        if (jsondata.libs === undefined) {
            this.libs = [];
        } else {
            this.libs = jsondata.libs;
        }
        if (jsondata.specialoutputs === undefined) {
            this.specialoutputs = [];
        } else {
            this.specialoutputs = jsondata.specialoutputs;
        }

        if (jsondata.tools === undefined) {
            this.tools = [];
        } else {
            this.tools = jsondata.tools;
        }
    }
}

export class ClientStateExecutor {
    compilerVisible = false;
    compilerOutputVisible = false;
    arguments: any[] = [];
    argumentsVisible = false;
    stdin = '';
    stdinVisible = false;
    compiler: ClientStateCompiler;
    wrap?: boolean;

    constructor(jsondata?) {
        if (jsondata) {
            // hack so TS doesn't think this.compiler is accessed before assignment below
            this.compiler = undefined as any as ClientStateCompiler;
            this.fromJsonData(jsondata);
        } else {
            this.compiler = new ClientStateCompiler();
            this.wrap = false;
        }

        delete this.compiler._internalid;
    }

    fromJsonData(jsondata) {
        if (jsondata.compilerVisible !== undefined) this.compilerVisible = jsondata.compilerVisible;
        if (jsondata.compilerOutputVisible !== undefined) this.compilerOutputVisible = jsondata.compilerOutputVisible;
        if (jsondata.arguments !== undefined) this.arguments = jsondata.arguments;
        if (jsondata.argumentsVisible !== undefined) this.argumentsVisible = jsondata.argumentsVisible;
        if (jsondata.stdin !== undefined) this.stdin = jsondata.stdin;
        if (jsondata.stdinVisible !== undefined) this.stdinVisible = jsondata.stdinVisible;
        if (jsondata.wrap !== undefined) this.wrap = jsondata.wrap;

        this.compiler = new ClientStateCompiler(jsondata.compiler);
    }
}

export class ClientStateConformanceView {
    libs: any[] = [];
    compilers: ClientStateCompiler[] = [];

    constructor(jsondata?) {
        if (jsondata) this.fromJsonData(jsondata);
    }

    fromJsonData(jsondata) {
        this.libs = jsondata.libs;
        if (jsondata.compilers) {
            for (const compilerdata of jsondata.compilers) {
                const compiler = new ClientStateCompiler(compilerdata);
                delete compiler._internalid;
                this.compilers.push(compiler);
            }
        }
    }
}

export class MultifileFile {
    id: any = undefined;
    fileId = 0;
    isIncluded = false;
    isOpen = false;
    isMainSource = false;
    filename: string | undefined = '';
    content = '';
    editorId = -1;
    langId = 'c++';

    constructor(jsondata?) {
        if (jsondata) this.fromJsonData(jsondata);
    }

    fromJsonData(jsondata) {
        this.fileId = jsondata.fileId;
        this.isIncluded = jsondata.isIncluded;
        this.isOpen = jsondata.isOpen;
        this.isMainSource = jsondata.isMainSource;
        this.filename = jsondata.filename;
        this.content = jsondata.content;
        this.editorId = jsondata.editorId;
        this.langId = jsondata.langId;
    }
}

export class ClientStateTree {
    id = 1;
    cmakeArgs = '';
    customOutputFilename = '';
    isCMakeProject = false;
    compilerLanguageId = 'c++';
    files: MultifileFile[] = [];
    newFileId = 1;
    compilers: ClientStateCompiler[] = [];
    executors: ClientStateExecutor[] = [];

    constructor(jsondata?) {
        if (jsondata) this.fromJsonData(jsondata);
    }

    fromJsonData(jsondata) {
        this.id = jsondata.id;
        this.cmakeArgs = jsondata.cmakeArgs;
        this.customOutputFilename = jsondata.customOutputFilename;
        this.isCMakeProject = jsondata.isCMakeProject;
        this.compilerLanguageId = jsondata.compilerLanguageId;

        let requiresFix = jsondata.newFileId === undefined;
        for (const file of jsondata.files) {
            const newFile = new MultifileFile(file);
            this.files.push(newFile);
            if (this.newFileId <= newFile.id) {
                requiresFix = true;
            }
        }

        if (requiresFix) {
            this.newFileId = 1;
            for (const file of this.files) {
                if (file.id > this.newFileId) {
                    this.newFileId = file.id + 1;
                }
            }
        } else {
            this.newFileId = jsondata.newFileId;
        }

        if (jsondata.compilers !== undefined) {
            for (const compilerdata of jsondata.compilers) {
                const compiler = new ClientStateCompiler(compilerdata);
                if (compiler.id) {
                    this.compilers.push(compiler);
                }
            }
        }

        if (jsondata.executors !== undefined) {
            for (const executor of jsondata.executors) {
                this.executors.push(new ClientStateExecutor(executor));
            }
        }
    }

    findOrCreateCompiler(id: number) {
        let foundCompiler;
        for (const compiler of this.compilers) {
            if (compiler._internalid === id) {
                foundCompiler = compiler;
            }
        }

        if (!foundCompiler) {
            foundCompiler = new ClientStateCompiler();
            foundCompiler._internalid = id;
            this.compilers.push(foundCompiler);
        }

        return foundCompiler;
    }
}

export class ClientStateSession {
    id: number | false = false;
    language = '';
    source = '';
    conformanceview: ClientStateConformanceView | false = false;
    compilers: any[] = [];
    executors: any[] = [];
    filename = undefined;

    constructor(jsondata?) {
        if (jsondata) this.fromJsonData(jsondata);
    }

    fromJsonData(jsondata) {
        if (jsondata.id !== undefined) this.id = jsondata.id;
        this.language = jsondata.language;
        this.source = jsondata.source;
        if (jsondata.conformanceview !== undefined) {
            if (jsondata.conformanceview) {
                this.conformanceview = new ClientStateConformanceView(jsondata.conformanceview);
            } else {
                this.conformanceview = false;
            }
        }

        for (const compilerdata of jsondata.compilers) {
            const compiler = new ClientStateCompiler(compilerdata);
            if (compiler.id) {
                this.compilers.push(compiler);
            }
        }

        if (jsondata.executors !== undefined) {
            for (const executor of jsondata.executors) {
                this.executors.push(new ClientStateExecutor(executor));
            }
        }

        if (jsondata.filename !== undefined) {
            this.filename = jsondata.filename;
        }
    }

    findOrCreateCompiler(id: number) {
        let foundCompiler;
        for (const compiler of this.compilers) {
            if (compiler._internalid === id) {
                foundCompiler = compiler;
            }
        }

        if (!foundCompiler) {
            foundCompiler = new ClientStateCompiler();
            foundCompiler._internalid = id;
            this.compilers.push(foundCompiler);
        }

        return foundCompiler;
    }

    countNumberOfSpecialOutputsAndTools() {
        let count = 0;

        if (this.conformanceview) count++;

        for (const compiler of this.compilers) {
            count += compiler.specialoutputs.length;
            if (compiler.tools) count += compiler.tools.length;
        }

        return count;
    }
}

export class ClientState {
    sessions: ClientStateSession[] = [];
    trees: ClientStateTree[] = [];

    constructor(jsondata?) {
        if (jsondata) this.fromJsonData(jsondata);
    }

    fromJsonData(jsondata) {
        for (const sessiondata of jsondata.sessions) {
            const session = new ClientStateSession(sessiondata);
            this.numberCompilersIfNeeded(session);
            this.sessions.push(session);
        }

        if (jsondata.trees) {
            for (const treedata of jsondata.trees) {
                this.trees.push(new ClientStateTree(treedata));
            }
        }
    }

    getNextCompilerId() {
        let nextId = 1;
        for (const session of this.sessions) {
            for (const compiler of session.compilers) {
                if (compiler._internalid && compiler._internalid >= nextId) {
                    nextId = compiler._internalid + 1;
                }
            }
        }
        return nextId;
    }

    numberCompilersIfNeeded(session: ClientStateSession, startAt?) {
        let id = startAt;
        let someIdsNeedNumbering = false;

        for (const compiler of session.compilers) {
            if (compiler._internalid) {
                if (compiler._internalid >= id) {
                    id = compiler._internalid + 1;
                }
            } else {
                someIdsNeedNumbering = true;
            }
        }

        if (someIdsNeedNumbering) {
            for (const compiler of session.compilers) {
                if (!compiler._internalid) {
                    compiler._internalid = id;
                    id++;
                }
            }
        }
    }

    findSessionById(id: number) {
        for (const session of this.sessions) {
            if (session.id === id) {
                return session;
            }
        }

        return false;
    }

    findTreeById(id: number) {
        for (let idxTree = 0; idxTree < this.trees.length; idxTree++) {
            const tree = this.trees[idxTree];

            if (tree.id === id) {
                return tree;
            }
        }

        return false;
    }

    findOrCreateSession(id: number) {
        let session = this.findSessionById(id);
        if (session) return session;

        session = new ClientStateSession();
        session.id = id;
        this.sessions.push(session);

        return session;
    }

    findOrCreateTree(id: number) {
        let tree = this.findTreeById(id);
        if (tree) return tree;

        tree = new ClientStateTree();
        tree.id = id;
        this.trees.push(tree);

        return tree;
    }
}
