/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <stdint.h>
#include <stdio.h>

extern "C" uint32_t __STACK_LIMIT;

int main() {
#if CORSTONE_TEST_MODE == 1
  printf("GUEST_EXIT_42\n");
  return 42;
#elif CORSTONE_TEST_MODE == 2
  printf("GUEST_TRIGGER_UDF\n");
  __asm__ volatile("udf #0");
  __builtin_unreachable();
#elif CORSTONE_TEST_MODE == 3
  printf("GUEST_TRIGGER_STKOF\n");
  // Set SP directly to __STACK_LIMIT and push below MSPLIM in a single inline
  // assembly block so compiler-generated SP-relative frame accesses cannot
  // interfere prior to the stack-limit fault.
  __asm__ volatile(
      "ldr r0, =__STACK_LIMIT\n"
      "mov sp, r0\n"
      "push {r0, r1}\n"
      :
      :
      : "r0", "memory");
  __builtin_unreachable();
#else
#error "Unsupported CORSTONE_TEST_MODE"
#endif
}
