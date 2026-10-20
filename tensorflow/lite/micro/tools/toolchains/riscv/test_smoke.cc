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
  printf("Hello from RISC-V RV32 bare-metal under QEMU!\n");

  volatile int32_t a = 14;
  volatile int32_t b = 3;

  // Test M extension (multiplication and division)
  volatile int32_t mul_res = a * b;
  volatile int32_t div_res = mul_res / b;

  printf("RV32IMC Math Check: %ld * %ld = %ld, %ld / %ld = %ld\n",
         static_cast<long>(a), static_cast<long>(b), static_cast<long>(mul_res),
         static_cast<long>(mul_res), static_cast<long>(b), static_cast<long>(div_res));

  if (mul_res == 42 && div_res == 14) {
    printf("SUCCESS: RISC-V RV32IMC smoke test passed!\n");
    return 0;
  } else {
    printf("FAILURE: Unexpected values mul=%ld, div=%ld\n",
           static_cast<long>(mul_res), static_cast<long>(div_res));
    return 1;
  }
}
