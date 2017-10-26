GHC?=ghc
demangle: demangle.hs
	$(GHC) -package ghc -dynamic demangle.hs -optl-Wl,-rpath,'$$ORIGIN' 
	ldd demangle \
		| sed -n -e 's/.*=> \([^ ]*\) .*/\1/p' \
		| egrep -v '^/lib' \
		| xargs cp -t .


clean:
	rm -f demangle.o demangle demangle.hi *.so*
