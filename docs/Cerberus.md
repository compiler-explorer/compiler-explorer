# Cerberus

[Cerberus](https://www.cl.cam.ac.uk/~pes20/cerberus/) offers executable semantics for a substantial fragment of the C
and CHERI-C languages. It is implemented via an elaboration into a simpler Core language, which is displayed as the
compiler output in the Compiler Explorer. Evaluation of C programs (execution) is also supported.

## Prerequisites

To build Cerberus, you need the **opam** package manager (>= 2.0.0, see [here](https://opam.ocaml.org/doc/Install.html)
for installation) and OCaml (>= 4.12.0).

Then, the following commands will set up the required opam repositories and download and install the required packages:

```sh
opam repo add --yes --this-switch coq-released https://coq.inria.fr/opam/released
opam repo add --yes --this-switch iris-dev https://gitlab.mpi-sws.org/iris/opam.git
opam pin --yes -n coq-struct-tact https://github.com/uwplse/StructTact.git
opam pin --yes -n coq-sail https://github.com/rems-project/coq-sail.git
opam pin --yes -n coq-cheri-capabilities https://github.com/rems-project/coq-cheri-capabilities.git
opam pin add -n --yes cerberus-lib https://github.com/rems-project/cerberus.git
opam pin add -n --yes cerberus https://github.com/rems-project/cerberus.git
opam pin add -n --yes cerberus-cheri https://github.com/rems-project/cerberus.git
opam install --yes cerberus cerberus-cheri
```

Now you can execute the `cerberus-cheri` and `cerberus` commands using `opam exec -- cerberus-cheri` or
`opam exec -- cerberus` respectively.

## Configuration

Once you have Cerberus installed, as described in the previous section, you can enable it by modifying the file
`etc/config/c.defaults.properties` as shown below. This change defines a group of two compilers: 'cerberus' for ISO C
and 'cerberus-cheri' for CHERI-C.

```
compilers=&gcc:&clang:&cerberus

group.cerberus.compilers=cerberus:cerberus-cheri
group.cerberus.compilerType=cerberus
group.cerberus.instructionSet=core
group.cerberus.demangler=
group.cerberus.postProcess=
group.cerberus.options=
group.cerberus.supportsBinary=false
group.cerberus.needsMulti=false
group.cerberus.supportsExecute=true
group.cerberus.interpreted=true
group.cerberus.versionFlag=--version

compiler.cerberus.name=Cerberus
compiler.cerberus.exe=cerberus
compiler.cerberus.objdumper=cerberus

compiler.cerberus-cheri.name=Cerberus-CHERI
compiler.cerberus-cheri.exe=cerberus-cheri
compiler.cerberus-cheri.objdumper=cerberus-cheri

```

## Limitations and Future Improvements

Presently, only simple Core output is shown. It is not syntactically highlighted nor linked to C source code locations.
Some potential future improvements include:

1. Error location handling in warning and error messages.
2. Specifying execution flags.
3. Core syntax highlighting.
4. Display of AST.
5. Display of other intermediate languages (Cabs, Ail).
6. Tooltips/links to the ISO C document from Core annotations.

## See also:

- [Cerberus (project page)](https://www.cl.cam.ac.uk/~pes20/cerberus/)
- [Cerberus (GitHub repository)](https://github.com/rems-project/cerberus)
- ["Formal Mechanised Semantics of CHERI C: Capabilities, Undefined Behaviour, and Provenance" (paper, preprint)](https://zaliva.org/cheric-asplos24.pdf)
- ["CHERI C semantics as an extension of the ISO C17 standard" (tech report)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-988.html)
