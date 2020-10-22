[![Build Status](https://travis-ci.org/compiler-explorer/compiler-explorer.svg?branch=master)](https://travis-ci.org/github/compiler-explorer/compiler-explorer)
[![codecov](https://codecov.io/gh/mattgodbolt/compiler-explorer/branch/master/graph/badge.svg)](https://codecov.io/gh/mattgodbolt/compiler-explorer)

![Compiler Explorer](docs/logo.svg)

Compiler Explorer
------------

**Compiler Explorer** is an interactive compiler. The left-hand pane shows
 editable C, C++, Rust, Go, D, Haskell, Swift, Pascal (and some more!) code.
The right, the assembly output of having compiled the code with a given
 compiler and settings. Multiple compilers are supported, and the UI layout
 is configurable (thanks to [GoldenLayout](https://www.golden-layout.com/)).
There is also an ispc compiler _[?](https://ispc.github.io/)_ for a C variant
 with extensions for SPMD.

Try out at [godbolt.org](https://godbolt.org).

You can support [this project on Patreon](https://patreon.com/mattgodbolt).

**Compiler Explorer** follows a [Code of Conduct](CODE_OF_CONDUCT.md) which
 aims to foster an open and welcoming environment.

**Compiler Explorer** was started in 2012 to serve my needs at [my previous employer](https://drw.com) to show how
 C++ constructs translated to assembly code. It started out as a `tmux` session with `vi` running in one
 pane and `watch gcc -S foo.cc -o -` running in the other.

Since then, it has become a public website serving around [2,000,000 compilations per week](https://www.stathat.com/cards/Tk5csAWI0O7x).

##### FAQ

There is now a FAQ section [in the repository wiki](https://github.com/compiler-explorer/compiler-explorer/wiki/FAQ).
 If your question is not present, please contact us as described below so we can help you.
 If you find that the FAQ is lacking some important point, please free to contribute to it and/or ask us to clarify it.

There are a number of videos that showcase some of the features of Compiler Explorer:
 A [presentation for CppCon 2019 about the project](https://www.youtube.com/watch?v=kIoZDUd5DKw)
 and an [older 2 part series of videos](https://www.youtube.com/watch?v=4_HL3PH4wDg) which go into a bit more detail
 into the more obscure features.

##### Contact us

We run a [Compiler Explorer Discord](https://discord.gg/B5WacA7), which is a place to discuss using or developing
 Compiler Explorer. We also have a presence on the [cpplang](https://cppalliance.org/slack/) slack channel
 `#compiler_explorer` and we have [a public mailing list](https://groups.google.com/forum/#!forum/compiler-explorer-discussion).

There's a development channel on the discord, and also a
 [development mailing list](https://groups.google.com/forum/#!forum/compiler-explorer-development).
 
Feel free to raise an issue on [github](https://github.com/compiler-explorer/compiler-explorer/issues) or
 [email Matt directly](mailto:matt@godbolt.org) for more help.

### Developing

**Compiler Explorer** is written in [Node.js](https://nodejs.org/).

Assuming you have a compatible version of `node` installed, simply running
 `make` ought to get you up and running with an Explorer running on port 10240
 on your local machine: http://localhost:10240/.
 Currently **Compiler Explorer**
 [requires the latest LTS](CONTRIBUTING.md#node-version) `node` version
 (_v12_) installed, either on the path or at `NODE_DIR`
 (an environment variable or `make` parameter).

Running with `make EXTRA_ARGS='--language LANG'` will allow you to load
 `LANG` exclusively, where `LANG` is one for the language ids/aliases defined
 in `lib/languages.js`. The `Makefile` will automatically install all the
 third party libraries needed to run; using `npm` to install server-side and
 client side components.

Some languages need extra tools to demangle them, e.g. `rust`, `d`, or `haskell`.
 Such tools are kept separately in the
 [tools repo](https://github.com/compiler-explorer/compiler-explorer-tools).

The config system leaves a lot to be desired.
 [Work has been done](https://github.com/rabsrincon/ccs-js) on porting
 [CCS](https://github.com/hellige/ccs-cpp) to Javascript and then something
 more rational can be used.


A [Road map](Roadmap.md) is available which gives a little insight into
 the future plans for **Compiler Explorer**.

### Running a local instance

If you want to point it at your own GCC or similar binaries, either edit the
 `etc/config/LANG.defaults.properties` or else make a new one with
 the name `LANG.local.properties`, substituting `LANG` as needed.
 `*.local.properties` files have the highest priority when loading properties.

When running in a corporate setting the URL shortening service can be replaced
 by an internal one if the default storage driver isn't appropriate for your
 environment. To do this, add a new module in `lib/shortener-myservice.js` and
 set the `urlShortenService` variable in configuration. This module should
 export a single function, see the [tinyurl module](lib/shortener-tinyurl.js)
 for an example.

### RESTful API

There's a simple restful API that can be used to do compiles to asm and to
 list compilers. 

You can find the API documentation [here](docs/API.md).

### Credits

**Compiler Explorer** is maintained by the awesome people listed in the
 [AUTHORS](AUTHORS.md) file.

We would like to thank the contributors listed in the
 [CONTRIBUTORS](CONTRIBUTORS.md) file, who have helped shape **Compiler Explorer**.

We would also like to specially thank these people for their contributions to
 **Compiler Explorer**:
- [Gabriel Devillers](https://github.com/voxelf)
 (_while working for [Kalray](http://www.kalrayinc.com/)_)
- [Johan Engelen](https://github.com/JohanEngelen)
- [Joshua Sheard](https://github.com/jsheard)
- [Marc Poulhiès](https://github.com/dkm)
- [Andrew Pardoe](https://github.com/AndrewPardoe)

A number of [amazing sponsors](https://godbolt.org/#sponsors), both individuals and companies, have helped fund and
 promote Compiler Explorer. We would also like to thank [JetBrains](https://www.jetbrains.com/) for their support
 and for donating licenses to their excellent products to develop **Compiler Explorer**.

![JetBrains](docs/jetbrains.svg)
