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
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ,
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
"use strict";

class ClientStateCompilerOptions {
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

class ClientStateCompiler {
    constructor(jsondata) {
        if (jsondata) {
            this.fromJsonData(jsondata);
        } else {
            this.id = "";
            this.options = "";
            this.filters = new ClientStateCompilerOptions();
            this.libs = [];
            this.specialoutputs = [];
            this.tools = [];
        }
    }

    fromJsonData(jsondata) {
        if (typeof jsondata.id !== 'undefined') {
            this.id = jsondata.id;
        } else if (typeof jsondata.compilerId !== 'undefined') {
            this.id = jsondata.compilerId;
        } else {
            this.id = "";
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

class ClientStateExecutor {
    constructor(jsondata) {
        if (jsondata) {
            this.fromJsonData(jsondata);
        } else {
            this.compiler = new ClientStateCompiler();
            this.compilerVisible = false;
            this.compilerOutputVisible = false;
            this.arguments = [];
            this.argumentsVisible = false;
            this.stdin = "";
            this.stdinVisible = false;
        }
    }

    fromJsonData(jsondata) {
        if (typeof jsondata.compilerVisible !== 'undefined')
            this.compilerVisible = jsondata.compilerVisible;
        if (typeof jsondata.compilerOutputVisible !== 'undefined')
            this.compilerOutputVisible = jsondata.compilerOutputVisible;
        if (typeof jsondata.arguments !== 'undefined')
            this.arguments = jsondata.arguments;
        if (typeof jsondata.argumentsVisible !== 'undefined')
            this.argumentsVisible = jsondata.argumentsVisible;
        if (typeof jsondata.stdin !== 'undefined')
            this.stdin = jsondata.stdin;
        if (typeof jsondata.stdinVisible !== 'undefined')
            this.stdinVisible = jsondata.stdinVisible;

        this.compiler = new ClientStateCompiler(jsondata.compiler);
    }
}

class ClientStateConformanceView {
    constructor(jsondata) {
        this.libs = [];
        this.compilers = [];

        if (jsondata) this.fromJsonData(jsondata);
    }

    fromJsonData(jsondata) {
        this.libs = jsondata.libs;
        jsondata.compilers.forEach((compilerdata) => {
            this.compilers.push(new ClientStateCompiler(compilerdata));
        });
    }
}

class ClientStateSession {
    constructor(jsondata) {
        this.id = false;
        this.language = "";
        this.source = "";
        this.conformanceview = false;
        this.compilers = [];
        this.executors = [];

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

        jsondata.compilers.forEach((compilerdata) => {
            this.compilers.push(new ClientStateCompiler(compilerdata));
        });

        if (typeof jsondata.executors !== 'undefined') {
            jsondata.executors.forEach((executor) => {
                this.executors.push(new ClientStateExecutor(executor));
            });
        }
    }

    findOrCreateCompiler(id) {
        let compiler = null;
        if (id <= this.compilers.length) {
            compiler = this.compilers[id - 1];
            return compiler;
        }

        for (let idx = this.compilers.length; idx < id; idx++) {
            compiler = new ClientStateCompiler();
            this.compilers.push(compiler);
        }

        return compiler;
    }

    countNumberOfSpecialOutputsAndTools() {
        let count = 0;

        if (this.conformanceview) count++;

        this.compilers.forEach((compiler) => {
            count += compiler.specialoutputs.length;
            if (compiler.tools) count += compiler.tools.length;
        });

        return count;
    }
}

class ClientState {
    constructor(jsondata) {
        this.sessions = [];

        if (jsondata) this.fromJsonData(jsondata);
    }

    fromJsonData(jsondata) {
        jsondata.sessions.forEach((sessiondata) => {
            this.sessions.push(new ClientStateSession(sessiondata));
        });
    }

    findSessionById(id) {
        let session = null;
        for (let idxSession = 0; idxSession < this.sessions.length; idxSession++) {
            session = this.sessions[idxSession];

            if (session.id === id) {
                return session;
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
}

module.exports = {
    State: ClientState,
    Session: ClientStateSession,
    Compiler: ClientStateCompiler,
    Executor: ClientStateExecutor,
    CompilerOptions: ClientStateCompilerOptions,
    ConformanceView: ClientStateConformanceView
};
