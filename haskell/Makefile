GHC?=ghc
demangle: demangle.hs
	$(GHC) -optl-static -optl-pthread -package ghc demangle.hs

clean:
	rm demangle.o demangle demangle.hi
