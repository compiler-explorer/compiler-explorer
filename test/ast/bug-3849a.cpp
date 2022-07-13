void f();

 char Bar(auto ) {
    return {};
}

int main() {
    char c = Bar([]{});
    return c;
}