import std.stdio;
import std.demangle;
import std.regex;

void main() {
    string dem(Captures!(string) m)
    {
        return demangle(m.hit);
    }

    foreach (line; stdin.byLine()) {
        string s2 = cast(string)line;
        auto s = replace!(dem)(s2, regex("[_$a-zA-Z][_$a-zA-Z0-9]*", "g"));
        writeln(s);
    }
}
