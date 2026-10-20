// tilelang target: {"kind":"c","tag":"","keys":["cpu"],"host":{"kind":"c","tag":"","keys":["cpu"]}}
#include <tl_templates/cpp/common.h>

#ifdef __cplusplus
extern "C"
#endif
int32_t add_kernel(half* A, half* B, half* C);
#ifdef __cplusplus
extern "C"
#endif
int32_t add_kernel(half* A, half* B, half* C) {
#line 6 "/app/example.py"
#line 12 "/app/example.py"
  for (int32_t i = 0; i < 8; ++i) {
#line 13 "/app/example.py"
    *(half8*)(C + (i * 8)) = (*(half8*)(A + (i * 8)) + *(half8*)(B + (i * 8)));
  }
  return 0;
}
