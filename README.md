[![Build Status](https://github.com/compiler-explorer/compiler-explorer/workflows/Compiler%20Explorer/badge.svg)](https://github.com/compiler-explorer/compiler-explorer/actions?query=workflow%3A%22Compiler+Explorer%22)
[![codecov](https://codecov.io/gh/compiler-explorer/compiler-explorer/branch/main/graph/badge.svg)](https://codecov.io/gh/compiler-explorer/compiler-explorer)

![Compiler Explorer](docs/logo.svg)

# Compiler Explorer

**Compiler Explorer** is an interactive compiler exploration website. Edit C, C++, Rust, Go, D, Haskell, Swift, Pascal, [ispc](https://ispc.github.io/) or other language code, and see how that code looks after being compiled in real time.
  Multiple compilers are supported, many different tools and visualations are available, and the UI layout
 is configurable (thanks to [GoldenLayout](https://www.golden-layout.com/)).

Try out at [godbolt.org](https://godbolt.org), or [run your own local instance](#running-a-local-instance).

You can financially support [this project on Patreon](https://patreon.com/mattgodbolt),
 [GitHub](https://github.com/sponsors/mattgodbolt/), [Paypal](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=KQWQZ7GPY2GZ6&item_name=Compiler+Explorer+development&currency_code=USD&source=url), or by
 buying cool gear on the [Compiler Explorer store](https://shop.spreadshirt.com/compiler-explorer/).

**Compiler Explorer** follows a [Code of Conduct](CODE_OF_CONDUCT.md) which
 aims to foster an open and welcoming environment.

**Compiler Explorer** was started in 2012 to show how C++ constructs translated to assembly code. It started out as a
 `tmux` session with `vi` running in one pane and `watch gcc -S foo.cc -o -` running in the other.

Since then, it has become a public website serving around [2,000,000 compilations per week](https://www.stathat.com/cards/Tk5csAWI0O7x).

## Using Compiler Explorer

### FAQ

There is now a FAQ section [in the repository wiki](https://github.com/compiler-explorer/compiler-explorer/wiki/FAQ).
 If your question is not present, please contact us as described below, so we can help you.
 If you find that the FAQ is lacking some important point, please free to contribute to it and/or ask us to clarify it.

### Videos

There are a number of videos that showcase some features of Compiler Explorer:

* [presentation for CppCon 2019 about the project](https://www.youtube.com/watch?v=kIoZDUd5DKw)
* [older 2 part series of videos](https://www.youtube.com/watch?v=4_HL3PH4wDg) which go into a bit more detail
 into the more obscure features.
* [playlist: Compiler Explorer](https://www.youtube.com/playlist?list=PL2HVqYf7If8dNYVN6ayjB06FPyhHCcnhG): A collection of videos discussing Compiler Explorer; using it, installing it, what it's for, etc.

## Developing

**Compiler Explorer** is written in [Node.js](https://nodejs.org/).

Assuming you have a compatible version of `node` installed, on Linux simply running
 `make` ought to get you up and running with an Explorer running on port 10240
 on your local machine: [http://localhost:10240/](http://localhost:10240/). If this doesn't work for you, please contact
 us, as we consider it important you can quickly and easily get running.
 Currently **Compiler Explorer**
 [requires the latest LTS](CONTRIBUTING.md#node-version) `node` version
 (_v12_) installed, either on the path or at `NODE_DIR`
 (an environment variable or `make` parameter).

Running with `make EXTRA_ARGS='--language LANG'` will allow you to load
 `LANG` exclusively, where `LANG` is one for the language ids/aliases defined
 in `lib/languages.js`. For example, to only run CE with C++ support, you'd run
 `make EXTRA_ARGS='--language c++'`. The `Makefile` will automatically install all the
 third party libraries needed to run; using `npm` to install server-side and
 client side components.

For development, we suggest using `make dev` to enable some useful features,
 such as automatic reloading on file changes and shorter startup times.

You can also use `npm run dev` to run if `make dev` doesn't work on your machine.

Some languages need extra tools to demangle them, e.g. `rust`, `d`, or `haskell`.
 Such tools are kept separately in the
 [tools repo](https://github.com/compiler-explorer/compiler-explorer-tools).

Configuring compiler explorer is achieved via configuration files in the `etc/config` directory. Values are
 `key=value`. Options in a `{type}.local.properties` file (where `{type}` is `c++` or similar) override anything in the
 `{type}.defaults.properties` file. There is a `.gitignore` file to ignore `*.local.*` files, so these won't be checked
 into git and you won't find yourself fighting with updated versions when you `git pull`. For more information see
 [Adding a Compiler](docs/AddingACompiler.md).

A [Road map](docs/Roadmap.md) is available which gives a little insight into
 the future plans for **Compiler Explorer**.

### Running a local instance

If you want to point it at your own GCC or similar binaries, either edit the
 `etc/config/LANG.defaults.properties` or else make a new one with
 the name `LANG.local.properties`, substituting `LANG` as needed.
 `*.local.properties` files have the highest priority when loading properties.

When running in a corporate setting the URL shortening service can be replaced
 by an internal one if the default storage driver isn't appropriate for your
 environment. To do this, add a new module in `lib/shortener/myservice.js` and
 set the `urlShortenService` variable in configuration. This module should
 export a single function, see the [tinyurl module](lib/shortener/tinyurl.js)
 for an example.

### RESTful API

There's a simple restful API that can be used to do compiles to asm and to
 list compilers. 

You can find the API documentation [here](docs/API.md).

## Contact us

We run a [Compiler Explorer Discord](https://discord.gg/B5WacA7), which is a place to discuss using or developing
Compiler Explorer. We also have a presence on the [cpplang](https://cppalliance.org/slack/) slack channel
`#compiler_explorer` and we have [a public mailing list](https://groups.google.com/forum/#!forum/compiler-explorer-discussion).

There's a development channel on the discord, and also a
[development mailing list](https://groups.google.com/forum/#!forum/compiler-explorer-development).

Feel free to raise an issue on [github](https://github.com/compiler-explorer/compiler-explorer/issues) or
[email Matt directly](mailto:matt@godbolt.org) for more help.

## Credits

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
 promote Compiler Explorer.
