[![Build Status](https://travis-ci.org/mattgodbolt/compiler-explorer.svg?branch=master)](https://travis-ci.org/mattgodbolt/compiler-explorer)
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

You can support this [this project on Patreon](https://patreon.com/mattgodbolt).

### Developing or running a local instance

Compiler Explorer is written in node.js.

Assuming you have npm and node installed, simply running `make` ought to get you up and running with an Explorer
running on port 10240 on your local machine: http://localhost:10240/

The `Makefile` will automatically install all the third party libraries needed to run; using `npm` to install server-side
components and `bower` to install all the client-facing libraries.

If you want to point it at your own GCC or similar binaries, either edit the `etc/config/compiler-explorer.defaults.properties` or else make a new one with the name
`compiler-explorer.local.properties`. `*.local.properties` files have the highest priority when loading properties.

The config system leaves a lot to be desired, I'm working on porting [CCS](https://github.com/hellige/ccs-cpp) to javascript and then something more rational can be used.

Feel free to raise an issue on [github](https://github.com/mattgodbolt/compiler-explorer/issues) or
[email me directly](mailto:matt@godbolt.org) for more help.

There's now a [Road map](Roadmap.md) that gives a little insight into future plans for Compiler Explorer.

### Credits

Compiler Explorer is maintained by [Matt Godbolt](http://xania.org) and [Rubén](https://github.com/RabsRincon).
Multiple compiler and difference view initially implemented by [Gabriel Devillers](https://github.com/voxelf),
while working for [Kalray](http://www.kalrayinc.com/).

### RESTful API

There's a simple restful API that can be used to do compiles to asm and to list compilers. In general
all handlers live in `/api/*` endpoints, and will accept JSON or text in POSTs, and will return text responses
or JSON responses depending on the request's `Accept` header.

At a later date there may be some form of rate-limiting: currently requests will be queued and dealt with
exactly like interactive requests on the main site. Authentication might be required at some point in the
future (for the main Compiler Explorer site anyway).

The following endpoints are defined:

#### `GET /api/compilers` - return a list of compilers

Returns a list of compilers. In text form, there's a simple formatting of the ID of the compiler and its
description. In JSON, all the information is returned as an array of compilers, with the `id` key being the
primary identifier of each compiler.

#### `POST /api/compiler/<compiler-id>/compile` - perform a compilation

To specify a compilation request as a JSON document, post it as the appropriate type and send an object of
the form: `{'source': 'source to compile', 'options': 'compiler flags', 'filters': {'filter': true}}`. The filters are an JSON object with true/false. If not supplied, defaults are used. If supplied, the filters are used
as-is.

A text compilation request has the source as the body of the post, and uses query parameters to pass the
options and filters. Filters are supplied as a comma-separated string. Use the query parameter `filters=XX`
to set the filters directly, else `addFilters=XX` to add a filter to defaults, or `removeFilters` to remove from defaults. Compiler parameters should be passed as `options=-O2` and default to empty.

Filters include `binary`, `labels`, `intel`, `comments` and `directives` and correspond to the UI buttons on
the HTML version.

The text request is designed for simplicity for command-line clients like `curl`:

```bash
$ curl 'https://gcc.godbolt.org/api/compiler/g63/compile?options=-Wall' --data-binary 'int foo() { return 1; }'
# Compilation provided by Compiler Explorer at gcc.godbolt.org
foo():
        push    rbp
        mov     rbp, rsp
        mov     eax, 1
        pop     rbp
        ret
```

If JSON is present in the request's `Accept` header, the compilation results are of the form:

```
{
    code: 0 if successful, else compiler return code,
    stdout: [ { text: "Output", 
                (optional) tag: {line: source line, text: "parsed error for that line"} } ],
    stderr: (as above),
    asm: [ { text: "assembly text", source: source line number or null if none } ]
}
```
