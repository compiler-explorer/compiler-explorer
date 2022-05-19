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

// grep -ro "eventHub\.on('.*'," static/ | cut -d "'" -f2 | sort | uniq
export type Events = {
    astViewClosed: [];
    astViewOpened: [];
    broadcastFontScale: [];
    cfgViewClosed: [];
    cfgViewOpened: [];
    colours: [];
    coloursForCompiler: [];
    coloursForEditor: [];
    compiler: [];
    compilerClose: [];
    compileResult: [];
    compilerFavoriteChange: [];
    compilerFlagsChange: [];
    compilerOpen: [];
    compilerSetDecorations: [];
    compiling: [];
    conformanceViewClose: [];
    conformanceViewOpen: [];
    copyShortLinkToClip: [];
    deviceViewClosed: [];
    deviceViewOpened: [];
    displaySharingPopover: [];
    editorChange: [];
    editorClose: [];
    editorLinkLine: [];
    editorOpen: [];
    editorSetDecoration: [];
    executeResult: [];
    executor: [];
    executorClose: [];
    executorOpen: [];
    filtersChange: [];
    findCompilers: [];
    findEditors: [];
    findExecutors: [];
    flagsViewClosed: [];
    flagsViewOpened: [];
    gccDumpFiltersChanged: [];
    gccDumpPassSelected: [];
    gccDumpUIInit: [];
    gccDumpViewClosed: [];
    gccDumpViewOpened: [];
    gnatDebugTreeViewClosed: [];
    gnatDebugTreeViewOpened: [];
    gnatDebugViewClosed: [];
    gnatDebugViewOpened: [];
    haskellCmmViewClosed: [];
    haskellCmmViewOpened: [];
    haskellCoreViewClosed: [];
    haskellCoreViewOpened: [];
    haskellStgViewClosed: [];
    haskellStgViewOpened: [];
    initialised: [];
    irViewClosed: [];
    irViewOpened: [];
    languageChange: [];
    modifySettings: [];
    motd: [];
    newSource: [];
    optViewClosed: [];
    optViewOpened: [];
    outputClosed: [];
    outputOpened: [];
    panesLinkLine: [];
    ppViewClosed: [];
    ppViewOpened: [];
    ppViewOptionsUpdated: [];
    requestCompilation: [];
    requestMotd: [];
    requestSettings: [];
    resendCompilation: [];
    resendExecution: [];
    resize: [];
    rustHirViewClosed: [];
    rustHirViewOpened: [];
    rustMacroExpViewClosed: [];
    rustMacroExpViewOpened: [];
    rustMirViewClosed: [];
    rustMirViewOpened: [];
    selectLine: [];
    settingsChange: [];
    setToolInput: [];
    shown: [];
    themeChange: [];
    toolClosed: [];
    toolInputChange: [];
    toolInputViewClosed: [];
    toolInputViewCloseRequest: [];
    toolOpened: [];
    toolSettingsChange: [];
    treeClose: [];
    treeCompilerEditorExcludeChange: [];
    treeCompilerEditorIncludeChange: [];
};
