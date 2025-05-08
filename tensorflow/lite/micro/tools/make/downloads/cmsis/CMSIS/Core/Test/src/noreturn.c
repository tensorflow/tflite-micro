// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

__NO_RETURN
static void func() {
    while(1);
}

void noreturn() {
    // CHECK-LABEL: <noreturn>:
    // CHECK: b 0x0 <noreturn>
    func();
    // CHECK-NOT: bx lr
}
