// REQUIRES: thumbv8m.base
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void get_sp_ns() {
    // CHECK-LABEL: <get_sp_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: mrs {{r[0-9]+}}, sp_ns
    volatile uint32_t result = __TZ_get_SP_NS();
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_sp_ns() {
    // CHECK-LABEL: <set_sp_ns>:
#if __ARM_FEATURE_CMSE == 3
    // CHECK-S: msr sp_ns, {{r[0-9]+}}
     __TZ_set_SP_NS(0x0815u);
#endif
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
