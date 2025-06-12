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

import adaDark from '../views/resources/logos/ada-dark.svg';
import ada from '../views/resources/logos/ada.svg';
import analysis from '../views/resources/logos/analysis.png';
import androidDark from '../views/resources/logos/android-dark.svg';
import android from '../views/resources/logos/android.svg';
import assembly from '../views/resources/logos/assembly.png';
import cpp from '../views/resources/logos/c++.svg';
import c from '../views/resources/logos/c.svg';
import c3 from '../views/resources/logos/c3.svg';
import camelia from '../views/resources/logos/camelia.svg';
import carbon from '../views/resources/logos/carbon.png';
import circt from '../views/resources/logos/circt.svg';
import clean from '../views/resources/logos/clean.svg';
import cmake from '../views/resources/logos/cmake.svg';
import crystalDark from '../views/resources/logos/crystal-dark.svg';
import crystal from '../views/resources/logos/crystal.svg';
import cudaDark from '../views/resources/logos/cuda-dark.svg';
import cuda from '../views/resources/logos/cuda.svg';
import d from '../views/resources/logos/d.svg';
import dart from '../views/resources/logos/dart.svg';
import dotnet from '../views/resources/logos/dotnet.svg';
import elixir from '../views/resources/logos/elixir.svg';
import erlang from '../views/resources/logos/erlang.svg';
import fortran from '../views/resources/logos/fortran.svg';
import fsharp from '../views/resources/logos/fsharp.svg';
import gimple from '../views/resources/logos/gimple.svg';
import glslDark from '../views/resources/logos/glsl-dark.svg';
import glsl from '../views/resources/logos/glsl.svg';
import go from '../views/resources/logos/go.svg';
import haskell from '../views/resources/logos/haskell.png';
import hlsl from '../views/resources/logos/hlsl.png';
import hookDark from '../views/resources/logos/hook-dark.png';
import hook from '../views/resources/logos/hook.png';
import hylo from '../views/resources/logos/hylo.svg';
import ispc from '../views/resources/logos/ispc.png';
import java from '../views/resources/logos/java.svg';
import js from '../views/resources/logos/js.svg';
import julia from '../views/resources/logos/julia.svg';
import kotlin from '../views/resources/logos/kotlin.svg';
import llvm from '../views/resources/logos/llvm.png';
import mlir from '../views/resources/logos/mlir.svg';
import mojo from '../views/resources/logos/mojo.svg';
import nim from '../views/resources/logos/nim.svg';
import nix from '../views/resources/logos/nix.svg';
import numba from '../views/resources/logos/numba.svg';
import ocaml from '../views/resources/logos/ocaml.svg';
import odin from '../views/resources/logos/odin.png';
import openclDark from '../views/resources/logos/opencl-dark.svg';
import opencl from '../views/resources/logos/opencl.svg';
import pascalDark from '../views/resources/logos/pascal-dark.svg';
import pascal from '../views/resources/logos/pascal.svg';
import pony from '../views/resources/logos/pony.svg';
import python from '../views/resources/logos/python.svg';
import racket from '../views/resources/logos/racket.svg';
import ruby from '../views/resources/logos/ruby.svg';
import rustDark from '../views/resources/logos/rust-dark.svg';
import rust from '../views/resources/logos/rust.svg';
import sail from '../views/resources/logos/sail.svg';
import scala from '../views/resources/logos/scala.png';
import slangDark from '../views/resources/logos/slang-dark.svg';
import slang from '../views/resources/logos/slang.svg';
import snowball from '../views/resources/logos/snowball.svg';
import solidity from '../views/resources/logos/solidity.svg';
import sonarDark from '../views/resources/logos/sonar-dark.svg';
import sonar from '../views/resources/logos/sonar.svg';
import spice from '../views/resources/logos/spice.png';
import spirvDark from '../views/resources/logos/spirv-dark.svg';
import spirv from '../views/resources/logos/spirv.svg';
import sway from '../views/resources/logos/sway.svg';
import swift from '../views/resources/logos/swift.svg';
import toit from '../views/resources/logos/toit.svg';
import ts from '../views/resources/logos/ts.svg';
import v from '../views/resources/logos/v.svg';
import vala from '../views/resources/logos/vala.svg';
import vyper from '../views/resources/logos/vyper.svg';
import wasm from '../views/resources/logos/wasm.svg';
import zig from '../views/resources/logos/zig.svg';

/** Get the logo base64 URL for an image */
export function getLogoImage(key: string | null): string | null {
    switch (key) {
        case null:
            return null;
        case 'ada.svg':
            return ada;
        case 'ada-dark.svg':
            return adaDark;
        case 'analysis.png':
            return analysis;
        case 'android.svg':
            return android;
        case 'android-dark.svg':
            return androidDark;
        case 'assembly.png':
            return assembly;
        case 'c++.svg':
            return cpp;
        case 'c.svg':
            return c;
        case 'c3.svg':
            return c3;
        case 'camelia.svg':
            return camelia;
        case 'carbon.png':
            return carbon;
        case 'circt.svg':
            return circt;
        case 'clean.svg':
            return clean;
        case 'cmake.svg':
            return cmake;
        case 'crystal.svg':
            return crystal;
        case 'crystal-dark.svg':
            return crystalDark;
        case 'cuda.svg':
            return cuda;
        case 'cuda-dark.svg':
            return cudaDark;
        case 'd.svg':
            return d;
        case 'dart.svg':
            return dart;
        case 'dotnet.svg':
            return dotnet;
        case 'elixir.svg':
            return elixir;
        case 'erlang.svg':
            return erlang;
        case 'fortran.svg':
            return fortran;
        case 'fsharp.svg':
            return fsharp;
        case 'gimple.svg':
            return gimple;
        case 'glsl.svg':
            return glsl;
        case 'glsl-dark.svg':
            return glslDark;
        case 'go.svg':
            return go;
        case 'haskell.png':
            return haskell;
        case 'hlsl.png':
            return hlsl;
        case 'hook.png':
            return hook;
        case 'hook-dark.png':
            return hookDark;
        case 'hylo.svg':
            return hylo;
        case 'ispc.png':
            return ispc;
        case 'java.svg':
            return java;
        case 'js.svg':
            return js;
        case 'julia.svg':
            return julia;
        case 'kotlin.svg':
            return kotlin;
        case 'llvm.png':
            return llvm;
        case 'mlir.svg':
            return mlir;
        case 'mojo.svg':
            return mojo;
        case 'nim.svg':
            return nim;
        case 'nix.svg':
            return nix;
        case 'numba.svg':
            return numba;
        case 'ocaml.svg':
            return ocaml;
        case 'odin.png':
            return odin;
        case 'opencl.svg':
            return opencl;
        case 'opencl-dark.svg':
            return openclDark;
        case 'pascal.svg':
            return pascal;
        case 'pascal-dark.svg':
            return pascalDark;
        case 'pony.svg':
            return pony;
        case 'python.svg':
            return python;
        case 'racket.svg':
            return racket;
        case 'ruby.svg':
            return ruby;
        case 'rust.svg':
            return rust;
        case 'rust-dark.svg':
            return rustDark;
        case 'sail.svg':
            return sail;
        case 'scala.png':
            return scala;
        case 'slang.svg':
            return slang;
        case 'slang-dark.svg':
            return slangDark;
        case 'snowball.svg':
            return snowball;
        case 'solidity.svg':
            return solidity;
        case 'sonar.svg':
            return sonar;
        case 'sonar-dark.svg':
            return sonarDark;
        case 'spice.png':
            return spice;
        case 'spirv.svg':
            return spirv;
        case 'spirv-dark.svg':
            return spirvDark;
        case 'sway.svg':
            return sway;
        case 'swift.svg':
            return swift;
        case 'toit.svg':
            return toit;
        case 'ts.svg':
            return ts;
        case 'v.svg':
            return v;
        case 'vala.svg':
            return vala;
        case 'vyper.svg':
            return vyper;
        case 'wasm.svg':
            return wasm;
        case 'zig.svg':
            return zig;
    }
    throw new Error(`Unknown logo key: ${key}`);
}
