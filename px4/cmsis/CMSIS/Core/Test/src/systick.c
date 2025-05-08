// REQUIRES: unsupported
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include <stdint.h>

typedef uint32_t IRQn_Type;
uint32_t SysTick_IRQn;

#include CORE_HEADER

void systick_type_ctrl() {
    // CHECK-LABEL: <systick_type_ctrl>:
    // CHECK: mov.w [[REG:r[0-9]+]], #0xe000e000
    // CHECK: ldr {{r[0-9]+}}, [[[REG]], #0x10]
    uint32_t ctrl = SysTick->CTRL;
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
