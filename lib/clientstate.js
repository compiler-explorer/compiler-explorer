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
    constructor(jsondata) {
        this.binary = false;
        this.commentOnly = true;
        this.demangle = true;
        this.directives = true;
        this.execute = false;
        this.intel = true;
        this.labels = true;
        this.trim = false;

        if (jsondata) this.fromJsonData(jsondata);
    }

    fromJsonData(jsondata) {
        if (typeof jsondata.binary !== 'undefined') this.binary = jsondata.binary;
        if (typeof jsondata.commentOnly !== 'undefined') this.commentOnly = jsondata.commentOnly;
        if (typeof jsondata.demangle !== 'undefined') this.demangle = jsondata.demangle;
        if (typeof jsondata.directives !== 'undefined') this.directives = jsondata.directives;
        if (typeof jsondata.execute !== 'undefined') this.execute = jsondata.execute;
        if (typeof jsondata.intel !== 'undefined') this.intel = jsondata.intel;
        if (typeof jsondata.labels !== 'undefined') this.labels = jsondata.labels;
        if (typeof jsondata.trim !== 'undefined') this.trim = jsondata.trim;
    }
}

export class ClientStateCompiler {
    constructor(jsondata) {
        this._internalid = undefined;

        if (jsondata) {
            this.fromJsonData(jsondata);
        } else {
            this.id = '';
            this.options = '';
            this.filters = new ClientStateCompilerOptions();
            this.libs = [];
            this.specialoutputs = [];
            this.tools = [];
        }
    }

    fromJsonData(jsondata) {
        if (typeof jsondata._internalid !== 'undefined') {
            this._internalid = jsondata._internalid;
        }

        if (typeof jsondata.id !== 'undefined') {
            this.id = jsondata.id;
        } else if (typeof jsondata.compilerId !== 'undefined') {
            this.id = jsondata.compilerId;
        } else {
            this.id = '';
        }

        this.options = jsondata.options;
        this.filters = new ClientStateCompilerOptions(jsondata.filters);
        if (typeof jsondata.libs !== 'undefined') {
            this.libs = jsondata.libs;
        } else {
            this.libs = [];
        }
        if (typeof jsondata.specialoutputs !== 'undefined') {
            this.specialoutputs = jsondata.specialoutputs;
        } else {
            this.specialoutputs = [];
        }

        if (typeof jsondata.tools !== 'undefined') {
            this.tools = jsondata.tools;
        } else {
            this.tools = [];
        }
    }
}

export class ClientStateExecutor {
    constructor(jsondata) {
        this.compilerVisible = false;
        this.compilerOutputVisible = false;
        this.arguments = [];
        this.argumentsVisible = false;
        this.stdin = '';
        this.stdinVisible = false;

        if (jsondata) {
            this.fromJsonData(jsondata);
        } else {
            this.compiler = new ClientStateCompiler();
            this.wrap = false;
        }

        delete this.compiler._internalid;
    }

    fromJsonData(jsondata) {
        if (typeof jsondata.compilerVisible !== 'undefined') this.compilerVisible = jsondata.compilerVisible;
        if (typeof jsondata.compilerOutputVisible !== 'undefined')
            this.compilerOutputVisible = jsondata.compilerOutputVisible;
        if (typeof jsondata.arguments !== 'undefined') this.arguments = jsondata.arguments;
        if (typeof jsondata.argumentsVisible !== 'undefined') this.argumentsVisible = jsondata.argumentsVisible;
        if (typeof jsondata.stdin !== 'undefined') this.stdin = jsondata.stdin;
        if (typeof jsondata.stdinVisible !== 'undefined') this.stdinVisible = jsondata.stdinVisible;
        if (typeof jsondata.wrap !== 'undefined') this.wrap = jsondata.wrap;

        this.compiler = new ClientStateCompiler(jsondata.compiler);
    }
}

export class ClientStateConformanceView {
    constructor(jsondata) {
        this.libs = [];
        this.compilers = [];

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
    constructor(jsondata) {
        this.fileId = 0;
        this.isIncluded = false;
        this.isOpen = false;
        this.isMainSource = false;
        this.filename = '';
        this.content = '';
        this.editorId = -1;
        this.langId = 'c++';

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
    constructor(jsondata) {
        this.id = 1;
        this.cmakeArgs = '';
        this.customOutputFilename = '';
        this.isCMakeProject = false;
        this.compilerLanguageId = 'c++';
        this.files = [];
        this.newFileId = 1;
        this.compilers = [];
        this.executors = [];

        if (jsondata) this.fromJsonData(jsondata);
    }

    fromJsonData(jsondata) {
        this.id = jsondata.id;
        this.cmakeArgs = jsondata.cmakeArgs;
        this.customOutputFilename = jsondata.customOutputFilename;
        this.isCMakeProject = jsondata.isCMakeProject;
        this.compilerLanguageId = jsondata.compilerLanguageId;

        let requiresFix = typeof jsondata.newFileId === 'undefined';
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

        if (typeof jsondata.compilers !== 'undefined') {
            for (const compilerdata of jsondata.compilers) {
                const compiler = new ClientStateCompiler(compilerdata);
                if (compiler.id) {
                    this.compilers.push(compiler);
                }
            }
        }

        if (typeof jsondata.executors !== 'undefined') {
            for (const executor of jsondata.executors) {
                this.executors.push(new ClientStateExecutor(executor));
            }
        }
    }

    findOrCreateCompiler(id) {
        let foundCompiler;
        for (let compiler of this.compilers) {
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
    constructor(jsondata) {
        this.id = false;
        this.language = '';
        this.source = '';
        this.conformanceview = false;
        this.compilers = [];
        this.executors = [];
        this.filename = undefined;

        if (jsondata) this.fromJsonData(jsondata);
    }

    fromJsonData(jsondata) {
        if (typeof jsondata.id !== 'undefined') this.id = jsondata.id;
        this.language = jsondata.language;
        this.source = jsondata.source;
        if (typeof jsondata.conformanceview !== 'undefined') {
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

        if (typeof jsondata.executors !== 'undefined') {
            for (const executor of jsondata.executors) {
                this.executors.push(new ClientStateExecutor(executor));
            }
        }

        if (typeof jsondata.filename !== 'undefined') {
            this.filename = jsondata.filename;
        }
    }

    findOrCreateCompiler(id) {
        let foundCompiler;
        for (let compiler of this.compilers) {
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
    constructor(jsondata) {
        this.sessions = [];
        this.trees = [];

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

    numberCompilersIfNeeded(session, startAt) {
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

    findSessionById(id) {
        for (const session of this.sessions) {
            if (session.id === id) {
                return session;
            }
        }

        return false;
    }

    findTreeById(id) {
        let tree = null;
        for (let idxTree = 0; idxTree < this.trees.length; idxTree++) {
            tree = this.trees[idxTree];

            if (tree.id === id) {
                return tree;
            }
        }

        return false;
    }

    findOrCreateSession(id) {
        let session = this.findSessionById(id);
        if (session) return session;

        session = new ClientStateSession();
        session.id = id;
        this.sessions.push(session);

        return session;
    }

    findOrCreateTree(id) {
        let tree = this.findTreeById(id);
        if (tree) return tree;

        tree = new ClientStateTree();
        tree.id = id;
        this.trees.push(tree);

        return tree;
    }
}
