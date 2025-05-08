// REQUIRES: thumb-2
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

static volatile uint32_t a = 10u;

void rbit() {
    // CHECK-LABEL: <rbit>:
    // CHECK: rbit {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t c = __RBIT(a);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

