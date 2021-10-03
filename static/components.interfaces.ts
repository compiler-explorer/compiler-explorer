// Copyright (c) 2021, Compiler Explorer Authors
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

import {Dictionary} from 'underscore';

export interface ComponentConfig<State extends ComponentState = ComponentState> {
    type: string;
    componentName: string;
    componentState: State;
}

export interface ComponentState {
    lang?: string;
}

export interface SourceComponentState extends ComponentState {
    source: number;
}

export interface TreeComponentState extends ComponentState {
    // The tree id
    tree: number;
}

export interface IdComponentState extends ComponentState {
    id: number;
}

// TODO: Add types
export interface OptionsComponentState extends ComponentState, SourceComponentState {
    options: any;
}

export interface LibraryItem {
    name: string;
    ver: string;
}

export interface CompilerComponentState extends ComponentState {
    compiler: string;
}

export interface LibraryComponentState extends CompilerComponentState, OptionsComponentState {
    libs: LibraryItem[];
}

export interface CompilerFilterComponentState extends LibraryComponentState {
    filters: Dictionary<boolean>;
}

export interface EditorComponentState extends CompilerComponentState, TreeComponentState {
    editor: number;
}

// TODO: Add types
export interface ToolIdComponentState extends ComponentState {
    toolId: any;
}

export interface ToolInputViewComponentState extends ToolIdComponentState {
    compilerId: string;
    toolName: string;
}

// TODO: Add types
export interface ToolViewComponentState extends EditorComponentState, ToolIdComponentState {
    args: string;
    monacoStdin: boolean;
}

export interface CompilerNameComponentState extends IdComponentState {
    compilerName: string;
}

// TODO: Add types
export interface EditorIdComponentState extends ComponentState {
    editorid: boolean;
}

// TODO: Add types
export interface OptimizationComponentState extends ComponentState {
    optOutput: any;
}

// TODO: Add types
export interface CompilerFlagsComponentState extends ComponentState, CompilerNameComponentState {
    compilerFlags: any;
}

// TODO: Add types
export interface AstOutputComponentState extends CompilerNameComponentState, SourceComponentState, EditorIdComponentState {
    astOutput: any;
}

export interface GccDumpViewComponentState extends ComponentState {
    _compilerid: number;
    _compilerName: string;
    _editorid: boolean;
}

// TODO: Add types
export interface GccDumpComponentConfig extends ComponentConfig<GccDumpViewComponentState> {
    treeDump?: any;
    rtlDump?: any;
    ipaDump?: any;
    addressOption?: any;
    slimOption?: any;
    rawOption?: any;
    detailsOption?: any;
    statsOption?: any;
    blocksOption?: any;
    vopsOption?: any;
    linenoOption?: any;
    uidOption?: any;
    allOption?: any;
    selectedPass?: string;
}

// TODO: Add types
export interface LangIdComponentState extends EditorIdComponentState, SourceComponentState {
    langId: any;
}

// TODO: Add types
export interface IrOutputComponentState extends CompilerNameComponentState, SourceComponentState, EditorIdComponentState {
    irOutput: any;
}

// TODO: Add types
export interface RustMirComponentState extends CompilerNameComponentState, SourceComponentState, EditorIdComponentState{
    rustMirOutput: any;
}

// TODO: Add types
export interface RustMacroComponentState extends CompilerNameComponentState, SourceComponentState, EditorIdComponentState {
    rustMacroExpOutput: any;
}

// TODO: Add types
export interface DeviceOutputComponentState extends CompilerNameComponentState, SourceComponentState, EditorIdComponentState {
    deviceOutput: any;
}
