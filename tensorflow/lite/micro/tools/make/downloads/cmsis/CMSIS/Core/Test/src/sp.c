// REQUIRES: armv7a
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

static volatile uint32_t u32;

void get_sp() {
    // CHECK-LABEL: <get_sp>:
    // CHECK: mov {{r[0-9]+}}, sp
    volatile uint32_t result = __get_SP();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_sp() {
    // CHECK-LABEL: <set_sp>:
    // CHECK: mov sp, {{r[0-9]+}}
    __set_SP(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_sp_usr() {
    // CHECK-LABEL: <get_sp_usr>:
    // CHECK: mrs [[REG:r[0-9]+]], apsr
    // CHECK: cps #{{31|0x1f}}
    // CHECK: mov {{r[0-9]+}}, sp
    // CHECK: msr CPSR_{{f?}}c, [[REG]]
    volatile uint32_t result = __get_SP_usr();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_sp_usr() {
    // CHECK-LABEL: <set_sp_usr>:
    // CHECK: mrs [[REG:r[0-9]+]], apsr
    // CHECK: cps #{{31|0x1f}}
    // CHECK: mov sp, {{r[0-9]+}}
    // CHECK: msr CPSR_{{f?}}c, [[REG]]
    __set_SP_usr(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
