// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void bkpt() {
    // CHECK-LABEL: <bkpt>:
    // CHECK: bkpt {{#0x15|#21}}
    __BKPT(0x15);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
