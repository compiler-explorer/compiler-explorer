GHC?=ghc
demangle: demangle.hs
	$(GHC) -package ghc -dynamic demangle.hs

clean:
	rm demangle.o demangle demangle.hi
