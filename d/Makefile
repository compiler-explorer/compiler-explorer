GDC?=gdc
DMD?=dmd
demangle: demangle.d
ifneq "" "$(shell which ${DMD})"
	$(DMD) -O -release $< -of$@
else
	$(GDC) -O2 $< -o $@
endif

clean:
	rm -f demangle.o demangle
