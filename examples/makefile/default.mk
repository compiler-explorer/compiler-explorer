# Recipe lines begin with a tab. CXX and CXXFLAGS are the compiler and options
# picked in Compiler Explorer, so this builds with whatever is selected above.
# output.s is the name Compiler Explorer looks for; build something else and put
# its name in the output file box instead.
output.s: example.cpp
	$(CXX) $(CXXFLAGS) -Wall -Wextra -o output.s example.cpp

clean:
	rm -f output.s
