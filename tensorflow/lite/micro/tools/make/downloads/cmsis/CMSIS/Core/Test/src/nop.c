// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void nop() {
    // CHECK-LABEL: <nop>:
    // CHECK: {{(nop|mov r8, r8)}}
    __NOP();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
