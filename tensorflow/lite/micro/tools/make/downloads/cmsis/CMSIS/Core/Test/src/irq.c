// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void enable_irq() {
    // CHECK-LABEL: <enable_irq>:
    // CHECK: cpsie i
    __enable_irq();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void disable_irq() {
    // CHECK-LABEL: <disable_irq>:
    // CHECK: cpsid i
    __disable_irq();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
