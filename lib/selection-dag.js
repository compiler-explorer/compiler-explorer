// Copyright (c) 2020, Dan Shechter
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
const _ = require('underscore'),
    path = require('path'),
    fs = require('fs-extra'),
    exec = require('./exec');

/*function isLLVMBased(compilerType, version) {
    return version.includes('clang') ||
        version.includes('LLVM') ||
        version.includes('rustc') ||
        compilerType === 'swift' ||
        compilerType === 'zig' ||
        compilerType === 'ispc';
}*/
async function generateSDAGStructure(baseCompiler, irArr) {
    const llcPath = path.join(path.dirname(baseCompiler.compiler.exe), 'llc');

    const irSrc = _.pluck(irArr, 'text')
        // Who doesn't like regex
        .map(function (x) { 
            return x.replace(/(, )?!dbg[^{\\n]*/g, "")
                .replace(/^.*call void @llvm.dbg.declare.*$/g, "")
                .replace(/^.*call void @llvm.dbg.value.*$/g, "");            
        }).join("\n");
    
    const dirPath = await baseCompiler.newTempDir();
    const inputFilename = path.join(dirPath, baseCompiler.compileFilename + ".ll");
    await fs.writeFile(inputFilename, irSrc);

    const options = [
        "-o",
        "/dev/null",
        "-view-dag-combine1-dags",
        "-view-legalize-types-dags",
        "-view-dag-combine-lt-dags",
        "-view-legalize-dags",
        "-view-dag-combine2-dags",
        "-view-isel-dags",
        "-view-sched-dags",
        //"-view-sunit-dags",

        inputFilename
    ];

    const execOptions = baseCompiler.getDefaultExecOptions();
    
    // Just give us the files
    execOptions.env["LLVM_GRAPHVIZ_VIEWER"] = "echo";

    const execResults = await
    exec.execute(llcPath, options,  execOptions);

    //const demangler = baseCompiler.demanglerClass ?
    //    new baseCompiler.demanglerClass(baseCompiler.compiler.demangler, baseCompiler) :
    //    undefined;
    
    const generatedDags = await Promise.all(
        _.without(execResults.stdout.split("\n"), "", null)
            .map(async function (x) {
                const elements = path.basename(x).split(".");
                return {
                    //func: demangler ? await demangler.process(elements[1]) : elements[1],
                    func: elements[1],
                    dagType: elements[2].substr(0, elements[2].lastIndexOf("-")),
                    graphvizData: await fs.readFile(x, 'utf8'),
                };
            }));
    
    const groupedByFunc = _.groupBy(generatedDags, 'func');
    return { dags: groupedByFunc};
}

module.exports.generateStructure = generateSDAGStructure;

