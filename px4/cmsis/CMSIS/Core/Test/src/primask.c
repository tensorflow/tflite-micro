// REQUIRES: thumbv6m
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void get_primask() {
    // CHECK-LABEL: <get_primask>:
    // CHECK: mrs {{r[0-9]+}}, primask
    volatile uint32_t result = __get_PRIMASK();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_primask_ns() {
    // CHECK-LABEL: <get_primask_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: mrs {{r[0-9]+}}, primask_ns
    volatile uint32_t result = __TZ_get_PRIMASK_NS();
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_primask() {
    // CHECK-LABEL: <set_primask>:
    // CHECK: msr primask, {{r[0-9]+}}
    __set_PRIMASK(0x0815u);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_primask_ns() {
    // CHECK-LABEL: <set_primask_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: msr primask_ns, {{r[0-9]+}}
     __TZ_set_PRIMASK_NS(0x0815u);
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
