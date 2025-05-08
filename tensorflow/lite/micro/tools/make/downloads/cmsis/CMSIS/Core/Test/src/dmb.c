// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void dmb() {
    // CHECK-LABEL: <dmb>:
    // CHECK: dmb sy
    __DMB();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

