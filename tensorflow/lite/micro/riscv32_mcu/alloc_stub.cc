/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstddef>
#include <cstdio>
#include <cstdlib>

// These stubs work around the issue of compiling code that uses
// the `new` and `delete` operators. The issue is that if the standard
// library included in the toolchain was compiled without -fno-exceptions,
// the linker will try to include exception handling sections and code
// in our binary, which we don't want, so we don't link against the standard
// C++ library at all... but then we don't have access to `new` and `delete`.

// operator new(size_t) should never return NULL
// (it should throw an exception instead), but programs an a microcontroller
// shouldn't use this operator at all, so we just hope for the best.
void* operator new(size_t size) { return malloc(size); }

void* operator new[](size_t size) { return malloc(size); }

void operator delete(void* p) noexcept {
  if (p) free(p);
}

void operator delete[](void* p) noexcept {
  if (p) free(p);
}

namespace std {

void __throw_bad_alloc(void) {
  puts("bad alloc");
  exit(1);
}

}  // namespace std
