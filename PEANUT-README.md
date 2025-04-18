# TFLite-Micro with Vector Intrinsics

This is Peanut Microsystems' fork of tflite-micro to optimize bottleneck operations using vector intrinsics.

## Building

Follow the guide in the *toolchains* repository for a guide on how to build and run *tflite-micro*. Instead of using the *riscv32_generic_makefile.inc*, use *riscv32_vector_makefile.inc* to build with vector intrinsics. Also, use the *rv32gcv* ISA for Spike. This is a superset of the instructions we intend to support.

To run with informative Peanut Microsystems-specific logs, add a PEANUT_MICRO_LOG flag in the PLATFORM_FLAGS of the *riscv32_vector_makefile.inc*:

    PLATFORM_FLAGS = \
        -march=$(RISCV_ARCH) \
        ... \
        -DPEANUT_MICRO_LOG

The main purpose for this flag is to sanity-check which implementations are used and to determine model architectures, including input and output shapes.
