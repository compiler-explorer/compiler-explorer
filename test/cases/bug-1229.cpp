#include <ctre.hpp>

static constexpr auto pattern = ctll::basic_fixed_string{ "h.*" };

constexpr auto match(std::string_view sv) noexcept {
    return ctre::re<pattern>().match(sv);
}

void myfunc() {
    match("hello");
}

