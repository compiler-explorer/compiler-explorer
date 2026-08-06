# Recipe lines begin with a tab. CXX and CXXFLAGS are the compiler and options
# picked in Compiler Explorer, so this builds with whatever is selected above.
output: example.cpp
	$(CXX) $(CXXFLAGS) -Wall -Wextra -o output example.cpp

clean:
	rm -f output
