// REQUIRES: thumbv6m
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void get_msp() {
    // CHECK-LABEL: <get_msp>:
    // CHECK: mrs {{r[0-9]+}}, msp
    volatile uint32_t result = __get_MSP();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_msp_ns() {
    // CHECK-LABEL: <get_msp_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: mrs {{r[0-9]+}}, msp_ns
    volatile uint32_t result = __TZ_get_MSP_NS();
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_msp() {
    // CHECK-LABEL: <set_msp>:
    // CHECK: msr msp, {{r[0-9]+}}
    __set_MSP(0x0815u);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_msp_ns() {
    // CHECK-LABEL: <set_msp_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: msr msp_ns, {{r[0-9]+}}
     __TZ_set_MSP_NS(0x0815u);
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
