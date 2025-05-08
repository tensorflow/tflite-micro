// REQUIRES: thumbv8m.main
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void get_msplim() {
    // CHECK-LABEL: <get_msplim>:
    // CHECK: mrs {{r[0-9]+}}, msplim
    volatile uint32_t result = __get_MSPLIM();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_msplim_ns() {
    // CHECK-LABEL: <get_msplim_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: mrs {{r[0-9]+}}, msplim_ns
    volatile uint32_t result = __TZ_get_MSPLIM_NS();
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_msplim() {
    // CHECK-LABEL: <set_msplim>:
    // CHECK: msr msplim, {{r[0-9]+}}
    __set_MSPLIM(0x0815u);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_msplim_ns() {
    // CHECK-LABEL: <set_msplim_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: msr msplim_ns, {{r[0-9]+}}
     __TZ_set_MSPLIM_NS(0x0815u);
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
