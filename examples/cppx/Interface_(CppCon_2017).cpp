
//====================================================================
// Library code: implementing the metaclass (once)

$class interface {
    constexpr {
        compiler.require($interface.variables().empty(),
                         "interfaces may not contain data");
        for... (auto f : $interface.functions()) {
            compiler.require(!f.is_copy() && !f.is_move(),
                "interfaces may not copy or move; consider a"
                " virtual clone() instead");
            if (!f.has_access()) f.make_public();
            compiler.require(f.is_public(),
                "interface functions must be public");
            f.make_pure_virtual();
        }
    }
    virtual ~interface() noexcept { }
};


//====================================================================
// User code: using the metaclass to write a type (many times)

interface Shape {
    int area() const;
    void scale_by(double factor);
};

    // try putting any of these lines into Shape to see "interface" rules
    // enforced => using the metaclass name to declare intent makes
    // this code more robust to such changes under maintenance
    //
    // int i;               // error: interfaces may not contain data
    // private: void g();   // error: interface functions must be public
    // Shape(const Shape&); // error: interfaces may not copy or move;
    //                      //        consider a virtual clone() instead

// Godbolt.org note: Click the "triangle ! icon" to see the output
constexpr {
    compiler.debug($Shape);
}


//====================================================================
// And then continue to use it as "just a class" as always... this is
// normal code just as if we'd written Shape not using a metaclass

class Circle : public Shape {
public:
    int area() const override { return 1; }
    void scale_by(double factor) override { }
};

#include <memory>

int main() {
    std::unique_ptr<Shape> shape = std::make_unique<Circle>();
    shape->area();
}
