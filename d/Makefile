GDC?=gdc
demangle: demangle.d
ifeq "" "$(shell which $DMD)"
	$(DMD) -O -release $< -of$@
else
	$(GDC) -O2 $< -o $@
endif

clean:
	rm -f demangle.o demangle
