ifeq (FreeBSD,$(shell uname -s))
    LDL :=
else
    LDL := -ldl
endif

all: libpreload.so

libpreload.so: preload.c
	$(CC) -std=c99 -shared -O1 -fPIC $^ -o $@ $(LDL)

.PHONY: test clean
test: libpreload.so
	-@rm -f /tmp/allowed
	cat tests/testfile | ./compiler-wrapper g++ -std=c++0x -S -o /tmp/allowed -x c++ -
	@if [ ! -s /tmp/allowed ]; then echo "/tmp/allowed should exist"; false; fi
	-@rm -f /tmp/allowed
	cat tests/bad-includes | ./compiler-wrapper g++ -std=c++0x -S -o /tmp/allowed -x c++ - 2>&1 | grep denied
	cat tests/bad-includes2 | ./compiler-wrapper g++ -std=c++0x -S -o /tmp/allowed -x c++ - 2>&1 | grep denied
	cat tests/bad-includes3 | ./compiler-wrapper g++ -std=c++0x -S -o /tmp/allowed -x c++ - 2>&1 | grep denied
	@if [ -s /tmp/allowed ]; then echo "/tmp/allowed should not exist"; false; fi
	-@rm -f not-allowed
	cat tests/testfile | ./compiler-wrapper g++ -std=c++0x -S -o not-allowed -x c++ - 2>&1 | grep denied
	@if [ -e not-allowed ]; then echo "not-allowed should not exist"; false; fi

clean:
	rm -f libpreload.so
