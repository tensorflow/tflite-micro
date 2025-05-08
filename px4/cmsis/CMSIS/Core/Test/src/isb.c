// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void isb() {
    // CHECK-LABEL: <isb>:
    // CHECK: isb sy
    __ISB();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

