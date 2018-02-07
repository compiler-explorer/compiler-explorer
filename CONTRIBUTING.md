# Contributing to Compiler Explorer

First of, if you're reading this: thank you! Even considering contributing to Compiler Explorer is very much appreciated!
Before we go too far, an apology: Compiler Explorer grew out of a bit of hacky JavaScript into a pretty large and
well-used project pretty quickly. Not all the code was originally well-written or well-tested. Please be forgiving of that, 
and be ready to help in improving that.

Note that Compiler Explorer targets the latest Node.js LTS, so it's better if you do so as well when testing your changes locally.
(But it *should* run in everything post-Node.js 8)

## In brief
* Make your changes, trying to stick to the style and code format where possible. We use default IntelliJ settings, 
  if that helps.
* If adding a new server-side component please do your best to add a test to cover it. For client-side changes that's trickier, 
  but do your best to help us improve this situation.
* Run the git hooks installer (Only needed once): `make install-git-hooks`. This will automatically run the tests before 
every commit, ensuring that they pass before commiting your changes. _You can disable this check with `git commit --no-verify` if needed_.
* Do a smoke test: run `make` and ensure the site works as you'd expect. Concentrate on the areas you'd expect to have
  changed, but if you can, click about generally to help check you haven't unintentionally broken something else.
* Submit a PR.
* If you have any questions, don't hesitate to email matt@godbolt.org or join the cpplang slack channel and talk on 
  channel #compiler_explorer

## Basic code layout

Code is separated into server-side code and client-side code. All dependencies (server and client side) are installed via `package.json`.
Server code is in `app.js` and in the `lib` directory. Client code is all in the `static` directory.

In the server code, the `app.js` sets up a basic `express` middleware-driven web server, delegating to the various compiler
backends in `lib/compilers`. Most inherit (loosely) from `lib/base-compiler.js` which does most of the work of running
compilers, then parsing the output and forming a JSON object to send to the client. Any assembly parsing is done in `asm.js`.

In the client code, GoldenLayout is used as the container. If you look at some of the components like the `static/compiler.js`,
you'll see the general flow. Any state stored makes it into the URL, so be careful not to stash anything too big in there.

The client code follows GoldenLayout's message-based system: no component has a reference to any other and everything is done
via messages. This will (in future) allow us to use pop-out windows, if we ever need to (as the messages are JSON-serialisable
between separate windows).

## Editing flow

The recommended way to work on Compiler Explorer is to just run `make` and let the automatic reload stuff do its magic.
Any changes to the server code will cause the server to reload, and any changes to the client code will be reflected upon
a page reload. This makes for a pretty quick turnaround.

## Gotchas

* Don't use new-style JS (`let` or arrow operators) in the client-side code. Sadly there's still enough users out there
  on old browsers. But feel free to use all the cool stuff on the server side code.
* Be aware Compiler Explorer runs on a cluster on the live site. No local state is kept between invocations, and it's likely
  the user's next request will hit another node in the cluster, so don't rely on any in-memory state.
