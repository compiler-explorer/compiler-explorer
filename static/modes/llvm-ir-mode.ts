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

import * as monaco from 'monaco-editor';

// This definition is based on the official LLVM vim syntax:
// http://llvm.org/svn/llvm-project/llvm/trunk/utils/vim/syntax/llvm.vim
// For VIM regex syntax, see: http://vimdoc.sourceforge.net/htmldoc/pattern.html
export function definition(): monaco.languages.IMonarchLanguage {
    return {
        // llvmType
        types: [
            'void',
            'half',
            'float',
            'double',
            'x86_fp80',
            'fp128',
            'ppc_fp128',
            'label',
            'metadata',
            'x86_mmx',
            'type',
            'label',
            'opaque',
            'token',
        ],
        // llvmStatement
        statements: [
            'add',
            'addrspacecast',
            'alloca',
            'and',
            'arcp',
            'ashr',
            'atomicrmw',
            'bitcast',
            'br',
            'catchpad',
            'catchswitch',
            'catchret',
            'call',
            'cleanuppad',
            'cleanupret',
            'cmpxchg',
            'eq',
            'exact',
            'extractelement',
            'extractvalue',
            'fadd',
            'fast',
            'fcmp',
            'fdiv',
            'fence',
            'fmul',
            'fpext',
            'fptosi',
            'fptoui',
            'fptrunc',
            'free',
            'frem',
            'fsub',
            'getelementptr',
            'icmp',
            'inbounds',
            'indirectbr',
            'insertelement',
            'insertvalue',
            'inttoptr',
            'invoke',
            'landingpad',
            'load',
            'lshr',
            'malloc',
            'max',
            'min',
            'mul',
            'nand',
            'ne',
            'ninf',
            'nnan',
            'nsw',
            'nsz',
            'nuw',
            'oeq',
            'oge',
            'ogt',
            'ole',
            'olt',
            'one',
            'or',
            'ord',
            'phi',
            'ptrtoint',
            'resume',
            'ret',
            'sdiv',
            'select',
            'sext',
            'sge',
            'sgt',
            'shl',
            'shufflevector',
            'sitofp',
            'sle',
            'slt',
            'srem',
            'store',
            'sub',
            'switch',
            'trunc',
            'udiv',
            'ueq',
            'uge',
            'ugt',
            'uitofp',
            'ule',
            'ult',
            'umax',
            'umin',
            'une',
            'uno',
            'unreachable',
            'unwind',
            'urem',
            'va_arg',
            'xchg',
            'xor',
            'zext',
        ],
        // llvmKeyword
        keywords: [
            'acq_rel',
            'acquire',
            'addrspace',
            'alias',
            'align',
            'alignstack',
            'alwaysinline',
            'appending',
            'argmemonly',
            'arm_aapcscc',
            'arm_aapcs_vfpcc',
            'arm_apcscc',
            'asm',
            'atomic',
            'available_externally',
            'blockaddress',
            'builtin',
            'byval',
            'c',
            'catch',
            'caller',
            'cc',
            'ccc',
            'cleanup',
            'coldcc',
            'comdat',
            'common',
            'constant',
            'datalayout',
            'declare',
            'default',
            'define',
            'deplibs',
            'dereferenceable',
            'distinct',
            'dllexport',
            'dllimport',
            'dso_local',
            'dso_preemptable',
            'except',
            'external',
            'externally_initialized',
            'extern_weak',
            'fastcc',
            'filter',
            'from',
            'gc',
            'global',
            'hhvmcc',
            'hhvm_ccc',
            'hidden',
            'initialexec',
            'inlinehint',
            'inreg',
            'inteldialect',
            'intel_ocl_bicc',
            'internal',
            'linkonce',
            'linkonce_odr',
            'localdynamic',
            'localexec',
            'local_unnamed_addr',
            'minsize',
            'module',
            'monotonic',
            'msp430_intrcc',
            'musttail',
            'naked',
            'nest',
            'noalias',
            'nobuiltin',
            'nocapture',
            'noimplicitfloat',
            'noinline',
            'nonlazybind',
            'nonnull',
            'norecurse',
            'noredzone',
            'noreturn',
            'nounwind',
            'optnone',
            'optsize',
            'personality',
            'private',
            'protected',
            'ptx_device',
            'ptx_kernel',
            'readnone',
            'readonly',
            'release',
            'returned',
            'returns_twice',
            'sanitize_address',
            'sanitize_memory',
            'sanitize_thread',
            'section',
            'seq_cst',
            'sideeffect',
            'signext',
            'syncscope',
            'source_filename',
            'speculatable',
            'spir_func',
            'spir_kernel',
            'sret',
            'ssp',
            'sspreq',
            'sspstrong',
            'strictfp',
            'swiftcc',
            'tail',
            'target',
            'thread_local',
            'to',
            'triple',
            'unnamed_addr',
            'unordered',
            'uselistorder',
            'uselistorder_bb',
            'uwtable',
            'volatile',
            'weak',
            'weak_odr',
            'within',
            'writeonly',
            'x86_64_sysvcc',
            'win64cc',
            'x86_fastcallcc',
            'x86_stdcallcc',
            'x86_thiscallcc',
            'zeroext',
        ],

        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        tokenizer: {
            root: [
                [/[,(){}<>[\]]/, 'delimiters'],
                [/i\d+\**/, 'type'], // llvmType

                // Misc syntax.
                [/[%@!]\d+/, 'variable.name'], // llvmNoName
                [/-?\d+\.\d*(e[+-]\d+)?/, 'number.float'], // llvmFloat
                [/0[xX][0-9A-Fa-f]+/, 'number.hex'], // llvmFloat
                [/-?\d+/, 'number'], // llvmNumber
                [/\b(true|false)\b/, 'keyword'], // llvmBoolean
                [/\b(zeroinitializer|undef|null|none)\b/, 'constant'], // llvmConstant
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-terminated string
                [/"/, 'string', '@string'], // push to string state
                [/[-a-zA-Z$._][-a-zA-Z$._0-9]*:/, 'tag'], // llvmLabel
                [/[%@][-a-zA-Z$._][-a-zA-Z$._0-9]*/, 'variable.name'], // llvmIdentifier

                // Named metadata and specialized metadata keywords.
                [/![-a-zA-Z$._][-a-zA-Z$._0-9]*(?=\s*)$/, 'identifier'], // llvmIdentifier
                [/![-a-zA-Z$._][-a-zA-Z$._0-9]*(?=\s*[=!])/, 'identifier'], // llvmIdentifier
                [/![A-Za-z]+(?=\s*\()/, 'type'], // llvmType
                [/\bDW_TAG_[a-z_]+\b/, 'constant'], // llvmConstant
                [/\bDW_ATE_[a-zA-Z_]+\b/, 'constant'], // llvmConstant
                [/\bDW_OP_[a-zA-Z0-9_]+\b/, 'constant'], // llvmConstant
                [/\bDW_LANG_[a-zA-Z0-9_]+\b/, 'constant'], // llvmConstant
                [/\bDW_VIRTUALITY_[a-z_]+\b/, 'constant'], // llvmConstant
                [/\bDIFlag[A-Za-z]+\b/, 'constant'], // llvmConstant

                // Syntax-highlight lit test commands and bug numbers.
                [/;\s*PR\d*\s*$/, 'comment.doc'], // llvmSpecialComment
                [/;\s*REQUIRES:.*$/, 'comment.doc'], // llvmSpecialComment
                [/;\s*RUN:.*$/, 'comment.doc'], // llvmSpecialComment
                [/;\s*CHECK:.*$/, 'comment.doc'], // llvmSpecialComment
                [/;\s*CHECK-(?:NEXT|NOT|DAG|SAME|LABEL):.*$/, 'comment.doc'], // llvmSpecialComment
                [/;\s*XFAIL:.*$/, 'comment.doc'], // llvmSpecialComment
                [/;.*$/, 'comment'],
                [/[*#=!]/, 'operators'],
                [
                    /[a-z_$][\w$]*/,
                    {
                        cases: {
                            '@statements': 'operators',
                            '@keywords': 'keyword',
                            '@types': 'type',
                            '@default': 'identifier',
                        },
                    },
                ],
                [/[ \t\r\n]+/, 'white'],
            ],
            string: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, 'string', '@pop'],
            ],
        },
    };
}

monaco.languages.register({id: 'llvm-ir'});
monaco.languages.setMonarchTokensProvider('llvm-ir', definition());

export {};
