// REQUIRES: clz
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

static volatile uint32_t a = 10u;

void clz() {
    // CHECK-LABEL: <clz>:
    // CHECK: clz {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t c = __CLZ(a);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

