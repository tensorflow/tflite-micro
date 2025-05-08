// REQUIRES: armv7a
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

static volatile uint32_t u32;

void get_cpsr() {
    // CHECK-LABEL: <get_cpsr>:
    // CHECK: mrs {{r[0-9]+}}, apsr
    volatile uint32_t result = __get_CPSR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_cpsr() {
    // CHECK-LABEL: <set_cpsr>:
    // CHECK: msr CPSR_fc, {{r[0-9]+}}
    __set_CPSR(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_mode() {
    // CHECK-LABEL: <get_mode>:
    // CHECK: mrs [[REG:r[0-9]+]], apsr
    // CHECK: and [[REG]], [[REG]], #{{31|0x1f}}
    volatile uint32_t result = __get_mode();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_mode() {
    // CHECK-LABEL: <set_mode>:
    // CHECK: msr CPSR_c, {{r[0-9]+}}
    __set_mode(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
