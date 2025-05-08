
// REQUIRES: thumbv6m
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void get_xpsr() {
    // CHECK-LABEL: <get_xpsr>:
    // CHECK: mrs {{r[0-9]+}}, xpsr
    volatile uint32_t result = __get_xPSR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
