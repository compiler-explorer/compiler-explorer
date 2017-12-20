//====================================================================
// Library code: implementing the metaclass (once)

$class basic_value {
    basic_value() = default;
    basic_value(const basic_value& that) = default;
    basic_value(basic_value&& that) = default;
    basic_value& operator=(const basic_value& that) = default;
    basic_value& operator=(basic_value&& that) = default;

    constexpr {
        for... (auto f : $basic_value.variables())
            if (!f.has_access()) f.make_private();
        for... (auto f : $basic_value.functions()) {
            if (!f.has_access()) f.make_public();
            compiler.require(!f.is_protected(), "a value type may not have a protected function");
            compiler.require(!f.is_virtual(),   "a value type may not have a virtual function");
            compiler.require(!f.is_destructor() || f.is_public(), "a value destructor must be public");
        }
    }
};

$class value : basic_value { };


//====================================================================
// User code: using the metaclass to write a type (many times)

value Point {
    int x = 0, y = 0;
    Point(int xx, int yy) : x{xx}, y{yy} { }
};

Point get_some_point() { return {1,1}; }

int main() {

    Point p1(50,100), p2;
    p2 = get_some_point();
    p2.x = 42;

}

// Compiler Explorer note: Click the "triangle ! icon" to see the output:
constexpr {
    compiler.debug($Point);
}
