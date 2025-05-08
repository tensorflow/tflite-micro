// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void sev() {
    // CHECK-LABEL: <sev>:
    // CHECK: sev
    __SEV();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

