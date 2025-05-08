// REQUIRES: thumbv6m
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void get_control() {
    // CHECK-LABEL: <get_control>:
    // CHECK: mrs {{r[0-9]+}}, control
    volatile uint32_t result = __get_CONTROL();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_control_ns() {
    // CHECK-LABEL: <get_control_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: mrs {{r[0-9]+}}, control_ns
    volatile uint32_t result = __TZ_get_CONTROL_NS();
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

volatile uint32_t v32 = 0x4711u;

void set_control() {
    // CHECK-LABEL: <set_control>:
    // CHECK: msr control, {{r[0-9]+}}
    __set_CONTROL(v32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_control_ns() {
    // CHECK-LABEL: <set_control_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: msr control_ns, {{r[0-9]+}}
    __TZ_set_CONTROL_NS(v32);
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

