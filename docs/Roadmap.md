# Compiler Explorer Road Map

This document is an attempt to capture thoughts on the future direction of Compiler Explorer. Last updated
December 2022. References to "I" and "me" here mean [Matt Godbolt](https://github.com/mattgodbolt).

## Areas to improve

### Code quality and testing

A project like Compiler Explorer thrives best when many people can easily contribute. There are many languages,
libraries, use-cases, and visualisations that the "core" team doesn't have experience with. In order to remain
supportable and keep serving various programming communities, we need to make the project easy to work with. We can
further improve code quality with more accurate typing, testing (increasing coverage, adding client tests), and
documentation to make it easier to on-board new contributors and keep it easy to support existing code.

Testing the website directly would also be an area to improve: we have the beginnings of some automated web browser
testing, but nothing stable or concrete yet.

## Considerations

### Tensions

There's an inherent tension between the standalone, run-it-yourself version of CE and the scalable, AWS-backed CE
instance. Care must be taken to keep the standalone version usable.

### Priorities

Above all, the priority is to keep the main CE site up, stable and dependable, free and accessible to as many people as
we can legally do so. That also means that URLs should live forever once they are created, which places a burden on us
to keep existing compilers and libraries available forever.

### Non-goals

Compiler Explorer will remain open-source and non-commercial. There are no plans at all to add "freemium" content. We do
have a Patreon site, GitHub sponsors, PayPal donations, and some corporate sponsors, for which we are incredibly
grateful. Funds from these source help support the cost of running the servers, save and plan for the future, and
incentivize the core team. Previously our goal was to remain "ad-free", that has been relaxed slightly to allow up to
three company sponsor logos visible at the top right of the screen.

## Goals

### 2023 goals

- **Modernising the codebase**. 2023 has to be the year of all Typescript. We need to continue pushing towards this goal
  as we are currently blocked using old versions of some libraries due to JS/TS/node.js/ES6 modules conflicts. We're
  still using an outdated and unsupported module loader and in order to move off it we need to update to a newer
  loader...that itself doesn't seemingly doesn't support a mixture of typescript and javascript like we currently have.
  In any case, moving to Typescript has improved our code quality and continues to find bugs, so we should absolutely
  port to it for this reason alone.
- **Improved compiler installation directory handling**. This project started in 2022 but left to rot a little, but our
  existing approach to sharing the 1700+ compilers and libraries between running instances is not scaling well. Behind
  the scenes we use NFS, and then for performance mount some compilers as read-only squashfs images from NFS. That
  worked well until we had 1000+ squashfs images to mount, one per compiler. A project to administrate and update a more
  layered squashfs image containing large numbers of unchanged compilers (think `docker` image layers) was started in
  2022, but fell behind due to so many other issues, and Matt being pretty rubbish at prioritising things. As nodes are
  now taking an awful long time to start up, this is becoming a problem.
- **Improved deployment flow**. The work to deploy Compiler Explorer has increased. At one point I couldn't get a site
  release out in the length of my train commute to work, which was a tipping point. I took a week off work in 2022 to
  fix that (amongst other things), but it's still a long and more-tedious-than-I-would-like process. With the recent
  addition of a GPU node, it's also more tricky than before to keep things updated. I'd like to find a way to get
  instances to start up without the need to pre-cache the compilation versions (a step we currently require to make
  startup fast enough). Ideally an update should be able to be directly applied to our staging setup from GitHub, and
  then promoted to production once things look good.
- **More resource pools**. Now we have two pools of servers - those with GPUs and those without - it would be great to
  support other server types too. For example, we could run our own Windows instances, or ARM-based instances too. This
  would allow us to natively run ARM code, or run `clang-cl` on a Windows instance we administrate and can make changes
  to. We probably need a better way of expressing the various pool types, and certainly this would need better
  deployment support (see above).

### Prior years' goals

#### 2022

Sadly I set no goals in 2022. I decided not to continue pursuing the login server stuff. It would be lovely but the
risks of data leakage/privacy violations, plus the work needed to get it running, seem to outweigh the benefits.

In lieu of having previously defined goals, I'd like to at least celebrate these achievements:

- UI improvements
  - The "Settings" panel has been cleaned up
  - Updated Control Flow Graph
  - Dark Theme+
  - Site Templates
- Library support for Rust
- 15 new languages, 600 new compilers across all lagnauges
- Execution on GPUs
- Excellent TypeScript conversion progress, making TypeScript the main language on the GitHub repo page

#### 2021

- **Login support**. A small amount of progress was made, but ultimately this was abandoned as other things seemed
  better use of the limited time I had available. I'd still like to do this at some point.
- **Multi-file support**. Thanks to [partouf](https://github.com/partouf) this landed! We can now compile and link
  multiple files.
- **Modernising the codebase**. Some progress, but still an awful lot of things need updating.
- **More compilers and libraries**. Almost everything installs via the new system.

#### 2020

**Was**: Finally tackle small screen device support.

**How did we do?**: a tiled, single-page read-only view was implemented. Our mobile support is about the best we could
hope for given the limitations of space and our current reliance on Microsoft's Monaco editor, which doesn't support
mobile at all.
