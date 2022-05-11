# Contributing to Compiler Explorer

First off, if you're reading this: thank you! Even considering contributing to **Compiler Explorer** is very much
appreciated! Before we go too far, an apology: **Compiler Explorer** grew out of a bit of hacky JavaScript into a pretty
large and well-used project pretty quickly. Not all the code was originally well-written or well-tested. Please be
forgiving of that.

**Compiler Explorer** follows a [Code of Conduct](CODE_OF_CONDUCT.md) which aims to foster an open and welcoming
environment.

# Where to start

We have labeled issues which should be easy to do that you can find
[here](https://github.com/compiler-explorer/compiler-explorer/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)

If you have any questions, don't hesitate: [Contact us].

If there is something you would like to do yourself, it might help to make an issue so people can weigh in and point you
in the right direction.

## Node version

**Compiler Explorer** currently targets [Node.js](https://nodejs.org/) LTS version 16 so it's better if you do so as
well when testing your changes locally.

## In brief

- Make your changes, trying to stick to the style and format where possible.
  - We use [ESLint](https://eslint.org/) to ensure a consistent code base and PRs won't pass unless it detects no
    errors.
  - Running `make lint` will run the linter, which will auto-fix everything it can and report back any errors and
    warnings.
- If you're adding a new server-side component, please do your best to add a test to cover it. For client-side changes
  that's trickier.
- Tests should run automatically as a pre-commit step. _You can disable this check with `git commit --no-verify` if
  needed_.
- You can run `make check` to run both the linter and the code tests
- Do a smoke test: Run `make` and ensure the site works as you'd expect. Concentrate on the areas you'd expect to have
  changed, but if you can, click about generally to help check you haven't unintentionally broken something else
- Submit a Pull Request.

## Basic code layout

Code is separated into server-side code and client-side code. All dependencies (server and client side) are installed
via `package.json`. _Server code_ is in `app.js` and in the `lib` directory. _Client code_ is all in the `static`
directory.

In the server code, the `app.js` sets up a basic `express` middleware-driven web server, delegating to the various
compiler backends in `lib/compilers/`. All of them inherit from `lib/base-compiler.js` which does most of the work of
running compilers, then parsing the output and forming a JSON object to send to the client. Any assembly parsing is done
in the `lib/asm-parser.js`, and similar, files.

In the client code, [GoldenLayout](https://www.golden-layout.com/) is used as the container. If you look at some
components like the `static/compiler.js`, you'll see the general flow. Any state stored makes it into the URL, so be
careful not to stash anything too big in there.

The client code follows GoldenLayout's message-based system: no component has a reference to any other and everything is
done via messages. This will allow us to use pop-out windows, if we ever need to, as the messages are JSON-serializable
between separate windows.

## Editing flow

The recommended way to work on **Compiler Explorer** is to just run `make dev` and let the automatic reloading do its
magic. Any changes to the server code will cause the server to reload, and any changes to the client code will be
reflected upon a page reload. This makes for a pretty quick turnaround. Note that a current issue makes every project
media asset to be locally unavailable. We will hopefully fix this in the near future.

## Gotchas

- New client-side code should preferably be written in TypeScript, but we will always accept js code too. Be aware that
  in that case, you must stick to **ES5** (so no `let` or arrow operators) js code. Sadly there are still enough users
  out there on old browsers. Note that this restriction does not apply to the server side code, in which you can use all
  the cool features you want. In lieu of ES6 features, [Underscore.js](https://underscorejs.org/) is available as a way
  to bridge the feature gap. The library is available both in the client and server code.
- Be aware that **Compiler Explorer** runs on a cluster on the live site. No local state is kept between invocations,
  and the user's next request will likely hit a different node in the cluster, so don't rely on any in-memory state.

[contact us]: README.md#contact-us
