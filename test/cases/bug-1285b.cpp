#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"

template<typename T>
void Ref(T&&);

#if 0
using std::string;
#else
// To make the disassembly easier to read, here's a string class that doesn't get inlined.
struct string {
  string() : size_(0), data_(nullptr) {}
  string(absl::string_view);
  string(string &&x) {
    size_ = x.size_;
    data_ = x.data_;
    x.size_ = 0;
    x.data_ = nullptr;
  }
  string(const string&);

  string& operator=(string &&x) {
    std::swap(size_, x.size_);
    std::swap(data_, x.data_);
    return *this;
  }
  //string& operator=(absl::string_view x);  // not in c++14 string

  ~string();

  string& assign(const char*, size_t);

  ptrdiff_t size_;
  char *data_;
};
#endif

//// END OF BOILERPLATE ////

class C {
  ::absl::optional<string> value_;
  void SetValue(::absl::optional<::absl::string_view> value);
};

void C::SetValue(::absl::optional<::absl::string_view> value) {
#if 0 // c++17's string has operator= for string_view, so just do this.
  value_ = value;
#else // In c++14, we have to do more work to avoid reallocation.
  if (value) {
    if (value_) {
      value_->assign(value->data(), value->size());
    } else {
      value_.emplace(*value);
    }
  } else {
    value_ = ::absl::nullopt;
  }
#endif
}
