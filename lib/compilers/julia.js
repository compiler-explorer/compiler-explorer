// Copyright (c) 2018, Elliot Saba
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

const BaseCompiler = require('../base-compiler')

class JuliaCompiler extends BaseCompiler {
    constructor(info, env) {
        super(info, env);
    }

    // No demangling for now
    postProcessAsm(result) {
        return result;
    }

    optionsForFilter(filters, outputFilename) {
        let optimize = "false";
        if (filters.optOutput) {
            optimize = "true";
        }
        return [
            "-e", "\
            using InteractiveUtils \n\
            m = Module(:Godbolt) \n\
            user_code = String(read(ARGS[end])) \n\
            Base.include_string(m, \"function main() \\n$user_code\\nend\") \n\
            out_file = ARGS[findfirst(ARGS .== \"-o\")+1] \n\
            open(out_file, \"w\") do io \n\
                InteractiveUtils.code_llvm(io, m.main, (); optimize="+optimize+") \n\
            end \n\
            exit(0)",
            // dummy value to force option parsing
            "dummy",
            // Set the output executable name
            "-o",
            outputFilename,
        ];
    }
}

module.exports = JuliaCompiler;
