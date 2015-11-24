In progress
-----------

Currently in progress is the difference view. In order to achieve this we need two "compiler" views; an "A" and "B". I'm in the progress of splitting out all the compiler-specifics into a class of sorts in the mainline, as this is a sensible refactor anyway. In the "diffs" branch I'm sketching out the UI.

Notes:

* storage in both hash tag and in local storage need to change. Specifically they are too tied to the notion of a single compiler.
* should ensure backwards compatibility: store off some hash tags and test the decode to the same view in the new scheme.
* binary mode could be done by compiling to .o (with LTO as necessary) and then decoding result of `objdump -d -C /tmp/temp.o -l -M intel --insn-width=16` (for example).


NB
--

Bug with clicked URLs and race to get the list of compilers (null compiler)
