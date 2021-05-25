# Compiler Explorer Road Map

This document is an attempt to capture thoughts on the future direction of Compiler Explorer. Last updated May 2021.

## Areas to improve

### Support more languages and compilers

A number of the open issues are to add more languages, libraries and compilers. Continuing to make it easier for others
to submit PRs to add new compilers is very important. This has improved, but not all compilers are installed using the
new approach. There's documentation on [adding a compiler](AddingACompiler.md),
[adding a new language](AddingALanguage.md)
and [adding a library](AddingALibrary.md).

### Multiple file support

As the site is getting used more and more, and in different ways to the initial intention, we would benefit from being
able to support multiple files in compilations. That is, C++ source files, and header files compiled together, or even
multiple C++ files compiled and linked together. This would let us showcase technologies like link-time optimization, or
C++ modules.

### Code quality and testing

A project like Compiler Explorer thrives best when many people can easily contribute. There are many languages,
libraries, use-cases, and visualisations that the "core" team doesn't have experience with. In order to remain
supportable and keep serving various programming communities, we need to make the project easy to work with. We can
improve code quality (e.g. move to transpiling from Typescript or similar to give us stronger types), testing
(increasing coverage, adding client tests), and documentation to make it easier to on-board new contributors and keep it
easy to support existing code.

## Considerations
### Tensions

There's an inherent tension between the standalone, run-it-yourself version of CE and the scalable, AWS-backed CE
instance. Care must be taken to keep the standalone version usable.

### Priorities

Above all, the priority is to keep the main CE site up, stable and dependable. That also means that URLs should live
forever once they are created, which places a burden on us to keep existing compilers and libraries available forever.

### Non-goals

Compiler Explorer will remain open-source and non-commercial. There are no plans at all to add "freemium" content. We do
have a Patreon site, Github sponsors, Paypal donations, and some corporate sponsors. Funds from these source help
support the cost of running the servers, and incentivize the core team. Previously our goal was to remain "ad-free",
that has been relaxed slightly to allow up to three company sponsor logos visible at the top right of the screen.

## Goals

### 2021 goals

* **Login support**. Support logging in to the site with GitHub, Google, etc. We will _never_ force you to log in for
  basic features, and of course will update the Privacy Policy. I won't be selling anything to do with user info etc,
  either: Logging in will be purely used to make _your_ life easier and allow you to manage things like shared settings
  and configuration, listing short URLs you've created (and potentially being able to remove them); and _maybe_ being
  able to make user-named short URLs (e.g. "godbolt.org/u/mattgodbolt/ctad-example"). This goal is a personal pet
  project of [Matt's](http://github.com/mattgodbolt/).
* **Multi-file support**. Multiple file compilation units to open the door to seeing LTO and maybe modules. This may
  include being able to use a `CMake` file to build things.
* **Modernising the codebase**. Moving the codebase to TypeScript, or something similar that will allow us to worry less
  about differences between front-end and back-end code (old Javascript versions), and help us attract more people to
  the project.
* **More compilers and libraries**. Plus finishing off the last stragglers of installation.


### Prior years' goals

#### 2020

**Was**: Finally tackle small screen device support.

**How did we do?**: a tiled, single-page read-only view was implemented. Our mobile support is about the best we could
hope for given the limitations of space and our current reliance on Microsoft's Monaco editor, which doesn't support
mobile at all.
