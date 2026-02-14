### **RISC-V 32-Bit Bare-Metal Support**

This target provides complete bare-metal support for the 32-bit RISC-V toolchain, including a UART driver and an automated QEMU emulation setup using `qemu-system-riscv32`.
It enables formatted output and testing on microcontroller-class RISC-V devices without relying on an operating system.

Note: Compatible with riscv32 baremetal toolchain configured with -march=rv32imac and -mabi=ilp32

**Building an Example:**

To build any example for the 'riscv32_baremetal' target, run:

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv32_baremetal <example_name>
```

This command performs cross-compilation for RISC-V and then automatically runs the binary under QEMU full-system emulation.

**Example:**

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv32_baremetal person_detection_test
```

**Expected Output:**

```
Testing TestInvoke
person data.  person score: 113, no person score: -113

no person data.  person score: -57, no person score: 57

Ran successfully

1/1 tests passed
~~~ALL TESTS PASSED~~~
```


