// REQUIRES: armv7a, fpu
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

static volatile uint32_t u32;

void get_fpexc() {
    // CHECK-LABEL: <get_fpexc>:
    // CHECK: vmrs {{r[0-9]+}}, fpexc
    volatile uint32_t result = __get_FPEXC();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_fpexc() {
    // CHECK-LABEL: <set_fpexc>:
    // CHECK: vmsr fpexc, {{r[0-9]+}}
    __set_FPEXC(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
