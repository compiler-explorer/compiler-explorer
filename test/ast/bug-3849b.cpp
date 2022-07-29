/*
#include "hell.hpp"

#pragma once

char Bar(auto ) {
    return {};
}

void f();
*/

int main() {
    char c = Bar([]{});
    return c;
}

#include <ctime>

clock_t myClock() {
  return clock();
}