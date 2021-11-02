#include <experimental/meta>
#include <experimental/compiler>

using namespace std::experimental;

//====================================================================
// Library code: implementing the metaclass (once)

consteval void interface(meta::info source) {
  for (meta::info mem : meta::member_range(source)) {
    meta::compiler.require(!meta::is_data_member(mem), "interfaces may not contain data");
    meta::compiler.require(!meta::is_copy(mem) && !meta::is_move(mem),
       "interfaces may not copy or move; consider"
       " a virtual clone() instead");

    if (meta::has_default_access(mem))
      meta::make_public(mem);

    meta::compiler.require(meta::is_public(mem), "interface functions must be public");

    meta::make_pure_virtual(mem);

    -> mem;
  }

  -> fragment struct X { virtual ~X() noexcept {} };
};


//====================================================================
// User code: using the metaclass to write a type (many times)

class(interface) Shape {
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

consteval {
  meta::compiler.print(reflexpr(Shape));
}

//====================================================================
// And then continue to use it as "just a class" as always... this is
// normal code just as if we'd written Shape not using a metaclass

class Circle : public Shape {
public:
    int area() const override { return 1; }
    void scale_by(double factor) override { }
};

consteval {
  meta::compiler.print(reflexpr(Circle));
}

#include <memory>

int main() {
    std::unique_ptr<Shape> shape = std::make_unique<Circle>();
    shape->area();
}
