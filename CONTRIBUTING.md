# Contributing to Compiler Explorer

First off, if you're reading this: thank you! Even considering contributing to
 **Compiler Explorer** is very much appreciated!
Before we go too far, an apology: **Compiler Explorer** grew out of a bit of
 hacky JavaScript into a pretty large and well-used project pretty quickly.
Not all the code was originally well-written or well-tested.
Please be forgiving of that, and be ready to help in improving that.

**Compiler Explorer** follows a [Code of Conduct](CODE_OF_CONDUCT.md) which
 aims to foster an open and welcoming environment.

## Node version
**Compiler Explorer** targets the latest [Node.js](https://nodejs.org/) LTS,
 so it's better if you do so as well when testing your changes locally.

Nonetheless, it _should_ run in everything post-Node.js 12.18. [Contact us] if
 this is not the case for you.

## In brief
* Make your changes, trying to stick to the style and format where possible.
    * We use [ESLint](https://eslint.org/) to ensure a consistent code base
    and PRs won't pass unless it detects no errors.
    * Running `make lint` will run the linter, which will auto-fix everything 
    it can and report back any errors and warnings.
* If adding a new server-side component please do your best to add a test to
 cover it. For client-side changes that's trickier.
* Run the git hooks installer (_Only needed once_): `make install-git-hooks`.
 This will automatically run the tests before you can commit, ensuring that 
 they pass before committing your changes.
 _You can disable this check with `git commit --no-verify` if needed_.
* You can run `make check` to run both the linter and the code tests
* Do a smoke test:
 Run `make` and ensure the site works as you'd expect. Concentrate on the
 areas you'd expect to have changed, but if you can, click about generally to
 help check you haven't unintentionally broken something else
* Submit a Pull Request.

If you have any questions, don't hesitate: [Contact us].

## Basic code layout

Code is separated into server-side code and client-side code.
All dependencies (server and client side) are installed via `package.json`.
_Server code_ is in `app.js` and in the `lib` directory. 
_Client code_ is all in the `static` directory.

In the server code, the `app.js` sets up a basic `express`
 middleware-driven web server, delegating to the various compiler backends in
 `lib/compilers/`. All of them inherit from `lib/base-compiler.js` which does
 most of the work of running compilers, then parsing the output and forming a
 JSON object to send to the client. Any assembly parsing is done in the
 `lib/asm-parser.js`, and similar, files.

In the client code, [GoldenLayout](https://www.golden-layout.com/) is used as
 the container. If you look at some of the components like the
 `static/compiler.js`, you'll see the general flow.
 Any state stored makes it into the URL, so be careful not to stash
 anything too big in there.

The client code follows GoldenLayout's message-based system:
 no component has a reference to any other and everything is done via messages.
 This will allow us to use pop-out windows, if we ever need to, as the messages
 are JSON-serializable between separate windows.

## Editing flow

The recommended way to work on **Compiler Explorer** is to just run `make dev`
 and let the automatic reloading do its magic.
Any changes to the server code will cause the server to reload, and any changes
 to the client code will be reflected upon a page reload.
 This makes for a pretty quick turnaround.
Note that a current issue makes every project media asset to be locally
 unavailable. We will hopefully fix this in the near future.

## Gotchas

* Stick to **ES5** (no `let` or arrow operators) in the client-side code.
 Sadly there are still enough users out there on old browsers,
 but feel free to use all the cool stuff on the server side code.
* Be aware that **Compiler Explorer** runs on a cluster on the live site.
 No local state is kept between invocations, and the user's next request will 
 likely hit a different node in the cluster, so don't rely on
 any in-memory state.

[Contact us]: README.md#contact-us

# Where to start

We have a project that lists a couple of issues that *Should* be easy to do @ https://github.com/compiler-explorer/compiler-explorer/projects/15 

If there is something you would like to do yourself, it might help to make an issue so people can weigh in and point you in the right direction.
