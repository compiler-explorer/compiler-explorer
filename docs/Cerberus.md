# Cerberus

[Cerberus](https://www.cl.cam.ac.uk/~pes20/cerberus/) offers executable semantics for a substantial fragment of C and
CHERI-C languages. It is implemented via an elaboration into a simpler Core language, which is displayed as the compiler
output in the Compiler Explorer. Evaluation of C programs (execution) is also supported.

## Prerequisites

The easiest way to install both the Cerberus and Cerberus-CHERI compilers is by using Docker:

`docker pull vzaliva/cerberus-cheri`

Then, for example, you can print the _Core_ elaboration for `test.c` using ISO C semantics:

`docker run -v $HOME/tmp:/mnt -it vzaliva/cerberus-cheri cerberus --pp=core --exec /mnt/test.c`

## Configuration

The file `etc/config/c.defaults.properties` defines a group of two compilers: 'cerberus' for ISO C and 'cerberus-cheri'
for CHERI-C. It assumes that the corresponding executables are in the path.

## Limitations and Future Improvement

Presently, only simple Core output is shown. It is not syntactically highlighted nor linked to C source code locations.
Some potential future improvements include:

1. Error location handling in warning and error messages
2. Specifying execution flags
3. Core syntax highlighting
4. Display of AST
5. Display of other intermediate languages (Cabs, Ail)
6. Tooltips/links to the ISO C document from Core annotations

## See also:

- [Cerberus (project page)](https://www.cl.cam.ac.uk/~pes20/cerberus/)
- [Cerberus (GitHub repository)](https://github.com/rems-project/cerberus)
- ["Formal Mechanised Semantics of CHERI C: Capabilities, Undefined Behaviour, and Provenance" (paper, preprint)](https://zaliva.org/cheric-asplos24.pdf)
- ["CHERI C semantics as an extension of the ISO C17 standard" (tech report)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-988.html)
