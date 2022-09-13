#include <eve/eve.hpp>

#include <eve/algo/for_each_iteration.hpp>
#include <eve/algo/preprocess_range.hpp>
#include <eve/algo/ptr_iterator.hpp>
#include <eve/algo/traits.hpp>

#include <eve/function/compress_store.hpp>
#include <eve/function/convert.hpp>
#include <eve/function/replace.hpp>

auto cvt(eve::logical<eve::wide<std::int16_t, eve::fixed<8>>> x) {
  using l_u = eve::logical<std::uint32_t>;
  return eve::convert(x, eve::as<l_u>{});
}