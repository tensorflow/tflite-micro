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

int main(int argc, char** argv) {
  printf("Hello from Xtensa bare-metal under xt-run simulator!\n");

  volatile int32_t a = 42;
  volatile int32_t b = 58;
  volatile int32_t sum = a + b;

  if (sum != 100) {
    printf("ALU check failed: %d + %d != 100\n", (int)a, (int)b);
    return 1;
  }

#if defined(HIFI3)
  printf("Core target: HiFi 3 / Fusion F1\n");
#elif defined(HIFI5)
  printf("Core target: HiFi 5\n");
#elif defined(VISION_P6)
  printf("Core target: Vision P6\n");
#endif

  printf("Xtensa smoke test passed!\n");
  return 0;
}
