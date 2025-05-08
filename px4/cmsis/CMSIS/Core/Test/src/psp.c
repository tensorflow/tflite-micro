// REQUIRES: thumbv6m
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void get_psp() {
    // CHECK-LABEL: <get_psp>:
    // CHECK: mrs {{r[0-9]+}}, psp
    volatile uint32_t result = __get_PSP();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_psp_ns() {
    // CHECK-LABEL: <get_psp_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: mrs {{r[0-9]+}}, psp_ns
    volatile uint32_t result = __TZ_get_PSP_NS();
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_psp() {
    // CHECK-LABEL: <set_psp>:
    // CHECK: msr psp, {{r[0-9]+}}
    __set_PSP(0x0815u);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_psp_ns() {
    // CHECK-LABEL: <set_psp_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: msr psp_ns, {{r[0-9]+}}
     __TZ_set_PSP_NS(0x0815u);
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
