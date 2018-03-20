[![Build Status](https://travis-ci.org/mattgodbolt/compiler-explorer.svg?branch=master)](https://travis-ci.org/mattgodbolt/compiler-explorer)
[![codecov](https://codecov.io/gh/mattgodbolt/compiler-explorer/branch/master/graph/badge.svg)](https://codecov.io/gh/mattgodbolt/compiler-explorer)

![Compiler Explorer](docs/logo.svg)

Compiler Explorer
------------

**Compiler Explorer** is an interactive compiler. The left-hand pane shows
 editable C, C++, Rust, Go, D, Haskell, Swift and Pascal code.
The right, the assembly output of having compiled the code with a given
 compiler and settings. Multiple compilers are supported, and the UI layout
 is configurable (thanks to [GoldenLayout](https://www.golden-layout.com/)).
There is also an ispc compiler _[?](https://ispc.github.io/)_ for a C variant
 with extensions for SPMD.

Try out at [godbolt.org](https://godbolt.org)

You can support [this project on Patreon](https://patreon.com/mattgodbolt).

**Compiler Explorer** follows a [Code of Conduct](CODE_OF_CONDUCT.md) which
 aims to foster an open and welcoming environment.

##### Contact us

For general discussion, feel free to join the mailing list at
 https://groups.google.com/forum/#!forum/compiler-explorer-discussion or the
 [cpplang](https://cpplang.now.sh/) slack channel `#compiler_explorer`.

If you are interested in developing, or want to see the discussions between
 existing developers, feel free to join the mailing list at
 https://groups.google.com/forum/#!forum/compiler-explorer-development or the
 [cpplang](https://cpplang.now.sh/) slack channel `#ce_implementation`.
 
Feel free to raise an issue on
 [github](https://github.com/mattgodbolt/compiler-explorer/issues) or
 [email Matt directly](mailto:matt@godbolt.org) for more help.

### Developing

**Compiler Explorer** is written in [Node.js](https://nodejs.org/).

Assuming you have a compatible version of `node` installed, simply running
 `make` ought to get you up and running with an Explorer running on port 10240
 on your local machine: http://localhost:10240/.
 Currently **Compiler Explorer**
 [requires the latest LTS](CONTRIBUTING.md#node-version) `node` version
 (_v8_) installed, either on the path or at `NODE_PATH`
 (an environment variable or `make` parameter).


Running with `make EXTRA_ARGS='--language LANG'` will allow you to load
 `LANG` exclusively, where `LANG` is one for the language ids/aliases defined
 in `lib/languages.js`. The `Makefile` will automatically install all the
 third party libraries needed to run; using `yarn` to install server-side and
 client side components.


The config system leaves a lot to be desired. Work has been done on porting
 [CCS](https://github.com/hellige/ccs-cpp) to Javascript and then something
 more rational can be used.


A [Road map](Roadmap.md) is available which gives a little insight into
 the future plans for **Compiler Explorer**.

### Running a local instance

If you want to point it at your own GCC or similar binaries, either edit the
 `etc/config/compiler-explorer.defaults.properties` or else make a new one with
 the name `compiler-explorer.local.properties`. `*.local.properties` files
 have the highest priority when loading properties.


When running in a corporate setting the URL shortening service can be replaced
 by an internal one to avoid leaking source code outside of the organization.
 This is done by adding a new module in `static/urlshorten-myservice.js` and
 setting the `urlShortenService` variable in configuration. This module should
 export a single function, see the [google module](static/urlshorten-google.js)
 for an example. `urlShortenService` can also be set to `none` to disable url
 shortening altogether.

### RESTful API

There's a simple restful API that can be used to do compiles to asm and to
 list compilers. In general all handlers live in `/api/*` endpoints, will
 accept JSON or text in POSTs, and will return text or JSON responses depending
 on the request's `Accept` header.

At a later date there may be some form of rate-limiting:
 currently, requests will be queued and dealt with in the same way interactive
 requests are done for the main site. Authentication might be required at some
 point in the future (for the main **Compiler Explorer** site anyway).

The following endpoints are defined:

#### `GET /api/languages` - return a list of languages

Returns a list of the currently supported languages, as pairs of languages IDs
 and their names.

#### `GET /api/compilers` - return a list of compilers

Returns a list of compilers. In text form, there's a simple formatting of the
 ID of the compiler, its description and its language ID. In JSON, all the
 information is returned as an array of compilers, with the `id` key being the
 primary identifier of each compiler.


#### `GET /api/compilers/<language-id>` - return a list of compilers with matching language

Returns a list of compilers for the provided language id. In text form,
 there's a simple formatting of the ID of the compiler, its description and its
 language ID. In JSON, all the information is returned as an array of compilers,
 with the `id` key being the primary identifier of each compiler.

#### `POST /api/compiler/<compiler-id>/compile` - perform a compilation

To specify a compilation request as a JSON document, post it as the appropriate
 type and send an object of the form:
```JSON
{
    "source": "Source to compile",
    "options": {
        "userArguments": "Compiler flags",
        "compilerOptions": {},
        "filters": {
            "filter": true
        }
    }
}
``` 
The filters are a JSON object with `true`/`false` values. If not supplied,
 defaults are used. If supplied, the filters are used as-is.
 The `compilerOptions` is used to pass extra arguments to the back end, and is
 probably not useful for most REST users.

A text compilation request has the source as the body of the post, and uses
 query parameters to pass the options and filters. Filters are supplied as a
 comma-separated string. Use the query parameter `filters=XX` to set the
 filters directly, else `addFilters=XX` to add a filter to defaults,
 or `removeFilters` to remove from defaults.
 Compiler parameters should be passed as `options=-O2` and default to empty.

Filters include `binary`, `labels`, `intel`, `comments`, `directives` and
 `demangle`, which correspond to the UI buttons on the HTML version.

The text request is designed for simplicity for command-line clients like `curl`

```bash
$ curl 'https://godbolt.org/api/compiler/g63/compile?options=-Wall' --data-binary 'int foo() { return 1; }'
# Compilation provided by Compiler Explorer at godbolt.org
foo():
        push    rbp
        mov     rbp, rsp
        mov     eax, 1
        pop     rbp
        ret
```

If JSON is present in the request's `Accept` header, the compilation results
 are of the form:

(_Optional values are marked with a `**`_)

```javascript
{
  "code": 0 if successful, else compiler return code,
  "stdout": [
            {
              "text": Output,
              ** "tag": {
                          "line": Source line,
                          "text": Parsed error for that line
                 }
            },
            ...
  ],
  "stderr": (format is similar to that of stdout),
  "asm": [
         {
           "text": Assembly text,
           "source": {file: null for user input, else path, line: number} or null if none
         },
         ...
  ],
  "okToCache": true if output could be locally cached else false,
  ** "optOutput" : {
                     "displayString" : String displayed in output,
                     "Pass" : [ Missed | Passed | Analysis ] (Specifies the type of optimisation output),
                     "Name" : Name of the output (mostly represents the reason for the output),
                     "DebugLoc" : {
                        "File": Name of file,
                        "Line": Line number,
                        "Column": Column number in line
                     },
                     "Function": Name of function for which optimisation output is provided,
                     "Args": Array of objects representing the arguments that the optimiser used when trying to optimise
     }
}
```

### Credits

**Compiler Explorer** is maintained by the awesome people listed in the 
 [AUTHORS](AUTHORS.md) file.

We would also like to specially thank these people for their contributions to
 **Compiler Explorer**:
- [Gabriel Devillers](https://github.com/voxelf)
 (_while working for [Kalray](http://www.kalrayinc.com/)_)
- [Johan Engelen](https://github.com/JohanEngelen)
- [Joshua Sheard](https://github.com/jsheard)
- [Marc Poulhiès](https://github.com/dkm)
- [Andrew Pardoe](https://github.com/AndrewPardoe)
