// Copyright (c) 2025, Compiler Explorer Authors
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

import adaIdeDark from '../views/resources/template_screenshots/Ada IDE.dark.png';
import adaIdeDarkplus from '../views/resources/template_screenshots/Ada IDE.darkplus.png';
import adaIdeDefault from '../views/resources/template_screenshots/Ada IDE.default.png';
import adaIdeOnedark from '../views/resources/template_screenshots/Ada IDE.onedark.png';
import adaIdePink from '../views/resources/template_screenshots/Ada IDE.pink.png';
import adaIdeRealDark from '../views/resources/template_screenshots/Ada IDE.real-dark.png';
import androidJavaIdeDark from '../views/resources/template_screenshots/Android Java IDE.dark.png';
import androidJavaIdeDarkplus from '../views/resources/template_screenshots/Android Java IDE.darkplus.png';
import androidJavaIdeDefault from '../views/resources/template_screenshots/Android Java IDE.default.png';
import androidJavaIdeOnedark from '../views/resources/template_screenshots/Android Java IDE.onedark.png';
import androidJavaIdePink from '../views/resources/template_screenshots/Android Java IDE.pink.png';
import androidJavaIdeRealDark from '../views/resources/template_screenshots/Android Java IDE.real-dark.png';
import androidKotlinIdeDark from '../views/resources/template_screenshots/Android Kotlin IDE.dark.png';
import androidKotlinIdeDarkplus from '../views/resources/template_screenshots/Android Kotlin IDE.darkplus.png';
import androidKotlinIdeDefault from '../views/resources/template_screenshots/Android Kotlin IDE.default.png';
import androidKotlinIdeOnedark from '../views/resources/template_screenshots/Android Kotlin IDE.onedark.png';
import androidKotlinIdePink from '../views/resources/template_screenshots/Android Kotlin IDE.pink.png';
import androidKotlinIdeRealDark from '../views/resources/template_screenshots/Android Kotlin IDE.real-dark.png';
import cppCmakeDark from '../views/resources/template_screenshots/C++ Cmake.dark.png';
import cppCmakeDarkplus from '../views/resources/template_screenshots/C++ Cmake.darkplus.png';
import cppCmakeDefault from '../views/resources/template_screenshots/C++ Cmake.default.png';
import cppCmakeOnedark from '../views/resources/template_screenshots/C++ Cmake.onedark.png';
import cppCmakePink from '../views/resources/template_screenshots/C++ Cmake.pink.png';
import cppCmakeRealDark from '../views/resources/template_screenshots/C++ Cmake.real-dark.png';
import diffOfTwoCompilersDark from '../views/resources/template_screenshots/Diff of Two Compilers.dark.png';
import diffOfTwoCompilersDarkplus from '../views/resources/template_screenshots/Diff of Two Compilers.darkplus.png';
import diffOfTwoCompilersDefault from '../views/resources/template_screenshots/Diff of Two Compilers.default.png';
import diffOfTwoCompilersOnedark from '../views/resources/template_screenshots/Diff of Two Compilers.onedark.png';
import diffOfTwoCompilersPink from '../views/resources/template_screenshots/Diff of Two Compilers.pink.png';
import diffOfTwoCompilersRealDark from '../views/resources/template_screenshots/Diff of Two Compilers.real-dark.png';
import diffOfTwoSourcesDark from '../views/resources/template_screenshots/Diff of Two Sources.dark.png';
import diffOfTwoSourcesDarkplus from '../views/resources/template_screenshots/Diff of Two Sources.darkplus.png';
import diffOfTwoSourcesDefault from '../views/resources/template_screenshots/Diff of Two Sources.default.png';
import diffOfTwoSourcesOnedark from '../views/resources/template_screenshots/Diff of Two Sources.onedark.png';
import diffOfTwoSourcesPink from '../views/resources/template_screenshots/Diff of Two Sources.pink.png';
import diffOfTwoSourcesRealDark from '../views/resources/template_screenshots/Diff of Two Sources.real-dark.png';
import javaIdeDark from '../views/resources/template_screenshots/Java IDE.dark.png';
import javaIdeDarkplus from '../views/resources/template_screenshots/Java IDE.darkplus.png';
import javaIdeDefault from '../views/resources/template_screenshots/Java IDE.default.png';
import javaIdeOnedark from '../views/resources/template_screenshots/Java IDE.onedark.png';
import javaIdePink from '../views/resources/template_screenshots/Java IDE.pink.png';
import javaIdeRealDark from '../views/resources/template_screenshots/Java IDE.real-dark.png';
import llvmIrDark from '../views/resources/template_screenshots/LLVM IR.dark.png';
import llvmIrDarkplus from '../views/resources/template_screenshots/LLVM IR.darkplus.png';
import llvmIrDefault from '../views/resources/template_screenshots/LLVM IR.default.png';
import llvmIrOnedark from '../views/resources/template_screenshots/LLVM IR.onedark.png';
import llvmIrPink from '../views/resources/template_screenshots/LLVM IR.pink.png';
import llvmIrRealDark from '../views/resources/template_screenshots/LLVM IR.real-dark.png';
import pascalIdeDark from '../views/resources/template_screenshots/Pascal IDE.dark.png';
import pascalIdeDarkplus from '../views/resources/template_screenshots/Pascal IDE.darkplus.png';
import pascalIdeDefault from '../views/resources/template_screenshots/Pascal IDE.default.png';
import pascalIdeOnedark from '../views/resources/template_screenshots/Pascal IDE.onedark.png';
import pascalIdePink from '../views/resources/template_screenshots/Pascal IDE.pink.png';
import pascalIdeRealDark from '../views/resources/template_screenshots/Pascal IDE.real-dark.png';
import preprocessorDark from '../views/resources/template_screenshots/Preprocessor.dark.png';
import preprocessorDarkplus from '../views/resources/template_screenshots/Preprocessor.darkplus.png';
import preprocessorDefault from '../views/resources/template_screenshots/Preprocessor.default.png';
import preprocessorOnedark from '../views/resources/template_screenshots/Preprocessor.onedark.png';
import preprocessorPink from '../views/resources/template_screenshots/Preprocessor.pink.png';
import preprocessorRealDark from '../views/resources/template_screenshots/Preprocessor.real-dark.png';

export function getScreenshotImage(key: string | null): string | null {
    switch (key) {
        case null:
            return null;
        case 'Ada IDE.darkplus.png':
            return adaIdeDarkplus;
        case 'Ada IDE.dark.png':
            return adaIdeDark;
        case 'Ada IDE.default.png':
            return adaIdeDefault;
        case 'Ada IDE.onedark.png':
            return adaIdeOnedark;
        case 'Ada IDE.pink.png':
            return adaIdePink;
        case 'Ada IDE.real-dark.png':
            return adaIdeRealDark;
        case 'Android Java IDE.darkplus.png':
            return androidJavaIdeDarkplus;
        case 'Android Java IDE.dark.png':
            return androidJavaIdeDark;
        case 'Android Java IDE.default.png':
            return androidJavaIdeDefault;
        case 'Android Java IDE.onedark.png':
            return androidJavaIdeOnedark;
        case 'Android Java IDE.pink.png':
            return androidJavaIdePink;
        case 'Android Java IDE.real-dark.png':
            return androidJavaIdeRealDark;
        case 'Android Kotlin IDE.darkplus.png':
            return androidKotlinIdeDarkplus;
        case 'Android Kotlin IDE.dark.png':
            return androidKotlinIdeDark;
        case 'Android Kotlin IDE.default.png':
            return androidKotlinIdeDefault;
        case 'Android Kotlin IDE.onedark.png':
            return androidKotlinIdeOnedark;
        case 'Android Kotlin IDE.pink.png':
            return androidKotlinIdePink;
        case 'Android Kotlin IDE.real-dark.png':
            return androidKotlinIdeRealDark;
        case 'C++ Cmake.darkplus.png':
            return cppCmakeDarkplus;
        case 'C++ Cmake.dark.png':
            return cppCmakeDark;
        case 'C++ Cmake.default.png':
            return cppCmakeDefault;
        case 'C++ Cmake.onedark.png':
            return cppCmakeOnedark;
        case 'C++ Cmake.pink.png':
            return cppCmakePink;
        case 'C++ Cmake.real-dark.png':
            return cppCmakeRealDark;
        case 'Diff of Two Compilers.darkplus.png':
            return diffOfTwoCompilersDarkplus;
        case 'Diff of Two Compilers.dark.png':
            return diffOfTwoCompilersDark;
        case 'Diff of Two Compilers.default.png':
            return diffOfTwoCompilersDefault;
        case 'Diff of Two Compilers.onedark.png':
            return diffOfTwoCompilersOnedark;
        case 'Diff of Two Compilers.pink.png':
            return diffOfTwoCompilersPink;
        case 'Diff of Two Compilers.real-dark.png':
            return diffOfTwoCompilersRealDark;
        case 'Diff of Two Sources.darkplus.png':
            return diffOfTwoSourcesDarkplus;
        case 'Diff of Two Sources.dark.png':
            return diffOfTwoSourcesDark;
        case 'Diff of Two Sources.default.png':
            return diffOfTwoSourcesDefault;
        case 'Diff of Two Sources.onedark.png':
            return diffOfTwoSourcesOnedark;
        case 'Diff of Two Sources.pink.png':
            return diffOfTwoSourcesPink;
        case 'Diff of Two Sources.real-dark.png':
            return diffOfTwoSourcesRealDark;
        case 'Java IDE.darkplus.png':
            return javaIdeDarkplus;
        case 'Java IDE.dark.png':
            return javaIdeDark;
        case 'Java IDE.default.png':
            return javaIdeDefault;
        case 'Java IDE.onedark.png':
            return javaIdeOnedark;
        case 'Java IDE.pink.png':
            return javaIdePink;
        case 'Java IDE.real-dark.png':
            return javaIdeRealDark;
        case 'LLVM IR.darkplus.png':
            return llvmIrDarkplus;
        case 'LLVM IR.dark.png':
            return llvmIrDark;
        case 'LLVM IR.default.png':
            return llvmIrDefault;
        case 'LLVM IR.onedark.png':
            return llvmIrOnedark;
        case 'LLVM IR.pink.png':
            return llvmIrPink;
        case 'LLVM IR.real-dark.png':
            return llvmIrRealDark;
        case 'Pascal IDE.darkplus.png':
            return pascalIdeDarkplus;
        case 'Pascal IDE.dark.png':
            return pascalIdeDark;
        case 'Pascal IDE.default.png':
            return pascalIdeDefault;
        case 'Pascal IDE.onedark.png':
            return pascalIdeOnedark;
        case 'Pascal IDE.pink.png':
            return pascalIdePink;
        case 'Pascal IDE.real-dark.png':
            return pascalIdeRealDark;
        case 'Preprocessor.darkplus.png':
            return preprocessorDarkplus;
        case 'Preprocessor.dark.png':
            return preprocessorDark;
        case 'Preprocessor.default.png':
            return preprocessorDefault;
        case 'Preprocessor.onedark.png':
            return preprocessorOnedark;
        case 'Preprocessor.pink.png':
            return preprocessorPink;
        case 'Preprocessor.real-dark.png':
            return preprocessorRealDark;
    }
    throw new Error(`Unknown screenshot key: ${key}`);
}
