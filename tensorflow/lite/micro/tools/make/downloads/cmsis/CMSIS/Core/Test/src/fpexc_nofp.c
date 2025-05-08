// REQUIRES: armv7a
// UNSUPPORTED: fpu
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

static volatile uint32_t u32;

void get_fpexc() {
    // CHECK-LABEL: <get_fpexc>:
    // CHECK-NOT: vmrs
    // CHECK: mov {{r[0-9]+}}, #0
    volatile uint32_t result = __get_FPEXC();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_fpexc() {
    // CHECK-LABEL: <set_fpexc>:
    // CHECK-NOT: vmsr
    __set_FPEXC(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
