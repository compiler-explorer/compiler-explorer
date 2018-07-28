// compile flags:
// [x86] cl /FA /EHsc /c vc-regex-example.cpp /Favc-regex.asm
// then, delete all non-regexTest lines
#include <string>
#include <regex>

void regexTest() {
     std::string s = "Some people,  when confronted with a problem, think "
           "\"I know, I'll use regular expressions.\" "
           "Now they have two problems.";
     auto self_regex = std::regex("REGULAR EXPRESSIONS",
             std::regex_constants::ECMAScript | std::regex_constants::icase);
}
