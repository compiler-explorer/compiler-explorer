[![Build Status](https://travis-ci.org/mattgodbolt/gcc-explorer.svg?branch=master)](https://travis-ci.org/mattgodbolt/gcc-explorer)
[![Codewake](https://www.codewake.com/badges/ask_question.svg)](https://www.codewake.com/p/compiler-explorer)

Compiler Explorer
------------

Compiler Explorer is an interactive compiler. The left-hand pane shows editable C/C++/Rust/Go/D code. The right, the
assembly output of having compiled the code with a given compiler and settings. Multiple compilers are supported, and
the UI layout is configurable (the [Golden Layout](https://www.golden-layout.com/) library is used for this).

Try out one of the demo sites: [C++][cpp], [Rust][rust], [D][d] or [Go][go].

[cpp]: https://gcc.godbolt.org/ "Compiler Explorer for C++"
[rust]: https://rust.godbolt.org/ "Compiler Explorer for Rust"
[d]: https://d.godbolt.org/ "Compiler Explorer for D"
[go]: https://go.godbolt.org/ "Compiler Explorer for Go"

### Developing

Compiler Explorer is written in node.js.

Assuming you have npm and node installed, simply running `make` ought to get you up and running with an Explorer
running on port 10240 on your local machine: http://localhost:10240/

The `Makefile` will automatically install all the third party libraries needed to run; using `npm` to install server-side
components and `bower` to install all the client-facing libraries.

If you want to point it at your own GCC or similar binaries, either edit the `etc/config/gcc-explorer.defaults.properties` or else make a new one with the name
`gcc-explorer.local.properties`. `*.local.properties` files have the highest priority when loading properties.

The config system leaves a lot to be desired, I'm working on porting [CCS](https://github.com/hellige/ccs-cpp) to javascript and then something more rational can be used.

Feel free to raise an issue on [github](https://github.com/mattgodbolt/gcc-explorer/issues) or
[email me directly](mailto:matt@godbolt.org) for more help.

### Credits

Compiler Explorer is maintained by [Matt Godbolt](http://xania.org). Multiple compiler and difference view was
implemented by [Gabriel Devillers](https://github.com/voxelf).
