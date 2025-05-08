// REQUIRES: thumb-2, thumbv7m
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void get_basepri() {
    // CHECK-LABEL: <get_basepri>:
    // CHECK: mrs {{r[0-9]+}}, basepri
    volatile uint32_t result = __get_BASEPRI();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_basepri_ns() {
    // CHECK-LABEL: <get_basepri_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: mrs {{r[0-9]+}}, basepri_ns
    volatile uint32_t result = __TZ_get_BASEPRI_NS();
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_basepri() {
    // CHECK-LABEL: <set_basepri>:
    // CHECK: msr basepri, {{r[0-9]+}}
    __set_BASEPRI(0x0815u);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_basepri_ns() {
    // CHECK-LABEL: <set_basepri_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: msr basepri_ns, {{r[0-9]+}}
     __TZ_set_BASEPRI_NS(0x0815u);
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_basepri_max() {
    // CHECK-LABEL: <set_basepri_max>:
    // CHECK: msr basepri_max, {{r[0-9]+}}
    __set_BASEPRI_MAX(0x0815u);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
