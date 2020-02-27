# Compiler Explorer Road Map
This document is an attempt to capture thoughts on the future direction of Compiler Explorer.

## Areas to improve

### Mobile client support

CE's UI doesn't work well with mobile clients. The [editor](https://github.com/Microsoft/monaco-editor)
 doesn't support mobile clients, and the layout doesn't lend itself well to small screens.

Ideas for improving mobile support include automatically folding up all the panes into a single tab upon
 detection of a mobile client. This would require a bunch of fixes in the
 underlying [UI library](http://golden-layout.com) as this doesn't properly work with mobile and tabs.

Perhaps a read-only simplified view would work better: the main reason one brings up the CE website is to
 look at tweeted links rather than author novel content. Note that there have been some tentative work
 on an app-based solution, but nothing has solidified yet.

### UI improvements

The UI has a number of things that need improving, but one of the things we are looking at is how to
 handle the loss of data that happens if one has a work-in-progress CE window open and then clicks another CE link.

### Support more compilers

Most of the open tickets are to do with adding new compilers, or fixing issues with existing compilers.
Continuing to add more compilers and make it easier for others to submit PRs to add new compilers is
very important. A [separate document](docs/AddingACompiler.md) explains how to add a compiler.

## Tensions

There's an inherent tension between the standalone, run-it-yourself version of CE and the scalable, AWS-backed
CE instance. Care must be taken to keep the standalone version usable.

## Priorities

Above all, the priority is to keep the main CE site up, stable and dependable.
That also means that URLs should live forever once they are created

## Non-goals

CE will remain ad-free, open-source and non-commercial. There's no plans at all to add "freemium" content,
despite the Patreon site where folks can help support the cost of running the servers.

## 2020 goals

A clear goal for 2020 is to finally tackle small screen device support.