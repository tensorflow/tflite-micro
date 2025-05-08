// REQUIRES: ldrex
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

void clrex() {
    // CHECK-LABEL: <clrex>:
    // CHECK: clrex
    __CLREX();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

