// REQUIRES: thumbv6m
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void get_apsr() {
    // CHECK-LABEL: <get_apsr>:
    // CHECK: mrs {{r[0-9]+}}, apsr
    volatile uint32_t result = __get_APSR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
