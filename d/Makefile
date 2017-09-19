GDC?=gdc
DMD?=dmd
LDC?=ldc2
demangle: demangle.d
ifneq "" "$(shell which ${DMD})"
	$(DMD) -O -release $< -of$@
else ifneq "" "$(shell which ${GDC})"
	$(GDC) -O2 $< -o $@
else
	$(LDC) -O2 $< -of=$@
endif

clean:
	rm -f demangle.o demangle
