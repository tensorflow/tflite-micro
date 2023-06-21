<!--ts-->
   * [TensorFlow Lite for Microcontrollers](#tensorflow-lite-for-microcontrollers)
   * [Build Status](#build-status)
      * [Official Builds](#official-builds)
      * [Community Supported TFLM Examples](#community-supported-tflm-examples)
      * [Community Supported Kernels and Unit Tests](#community-supported-kernels-and-unit-tests)
   * [Contributing](#contributing)
   * [Getting Help](#getting-help)
   * [Additional Documentation](#additional-documentation)
   * [RFCs](#rfcs)

<!-- Added by: advaitjain, at: Mon 04 Oct 2021 11:23:57 AM PDT -->

<!--te-->

# TensorFlow Lite for Microcontrollers

TensorFlow Lite for Microcontrollers is a port of TensorFlow Lite designed to
run machine learning models on DSPs, microcontrollers and other devices with
limited memory.

Additional Links:
 * [Tensorflow github repository](https://github.com/tensorflow/tensorflow/)
 * [TFLM at tensorflow.org](https://www.tensorflow.org/lite/microcontrollers)

# Build Status

 * [GitHub Status](https://www.githubstatus.com/)

## Official Builds

Build Type       |    Status     |
-----------      | --------------|
CI (Linux)       | [![CI](https://github.com/tensorflow/tflite-micro/actions/workflows/run_ci.yml/badge.svg)](https://github.com/tensorflow/tflite-micro/actions/workflows/run_ci.yml) |
Code Sync        | [![Sync from Upstream TF](https://github.com/tensorflow/tflite-micro/actions/workflows/sync.yml/badge.svg)](https://github.com/tensorflow/tflite-micro/actions/workflows/sync.yml) |


## Community Supported TFLM Examples
This table captures platforms that TFLM has been ported to. Please see
[New Platform Support](tensorflow/lite/micro/docs/new_platform_support.md) for
additional documentation.

Platform      |    Status     |
-----------     | --------------|
Arduino         | [![Arduino](https://github.com/tensorflow/tflite-micro-arduino-examples/actions/workflows/ci.yml/badge.svg)](https://github.com/tensorflow/tflite-micro-arduino-examples/actions/workflows/ci.yml) [![Antmicro](https://github.com/antmicro/tensorflow-arduino-examples/actions/workflows/test_examples.yml/badge.svg)](https://github.com/antmicro/tensorflow-arduino-examples/actions/workflows/test_examples.yml) |
[Coral Dev Board Micro](https://coral.ai/products/dev-board-micro) | [TFLM + EdgeTPU Examples for Coral Dev Board Micro](https://github.com/google-coral/coralmicro) |
Espressif Systems Dev Boards  | [![ESP Dev Boards](https://github.com/espressif/tflite-micro-esp-examples/actions/workflows/ci.yml/badge.svg)](https://github.com/espressif/tflite-micro-esp-examples/actions/workflows/ci.yml) |
Renesas Boards | [TFLM Examples for Renesas Boards](https://github.com/renesas/tflite-micro-renesas) |
Silicon Labs Dev Kits        | [TFLM Examples for Silicon Labs Dev Kits](https://github.com/SiliconLabs/tflite-micro-efr32-examples)
Sparkfun Edge   | [![Sparkfun Edge](https://github.com/advaitjain/tflite-micro-sparkfun-edge-examples/actions/workflows/ci.yml/badge.svg?event=schedule)](https://github.com/advaitjain/tflite-micro-sparkfun-edge-examples/actions/workflows/ci.yml)
Texas Instruments Dev Boards | [![Texas Instruments Dev Boards](https://github.com/TexasInstruments/tensorflow-lite-micro-examples/actions/workflows/ci.yml/badge.svg?event=status)](https://github.com/TexasInstruments/tensorflow-lite-micro-examples/actions/workflows/ci.yml)


## Community Supported Kernels and Unit Tests
This is a list of targets that have optimized kernel implementations and/or run
the TFLM unit tests using software emulation or instruction set simulators.

Build Type      |    Status     |
-----------     | --------------|
Cortex-M        | [![Cortex-M](https://github.com/tensorflow/tflite-micro/actions/workflows/cortex_m.yml/badge.svg)](https://github.com/tensorflow/tflite-micro/actions/workflows/cortex_m.yml) |
Hexagon         | [![Hexagon](https://github.com/tensorflow/tflite-micro/actions/workflows/run_hexagon.yml/badge.svg)](https://github.com/tensorflow/tflite-micro/actions/workflows/run_hexagon.yml) |
RISC-V          | [![RISC-V](https://github.com/tensorflow/tflite-micro/actions/workflows/riscv.yml/badge.svg)](https://github.com/tensorflow/tflite-micro/actions/workflows/riscv.yml) |
Xtensa          | [![Xtensa](https://github.com/tensorflow/tflite-micro/actions/workflows/run_xtensa.yml/badge.svg)](https://github.com/tensorflow/tflite-micro/actions/workflows/run_xtensa.yml) |
Generate Integration Test          | [![Generate Integration Test](https://github.com/tensorflow/tflite-micro/actions/workflows/generate_integration_tests.yml/badge.svg)](https://github.com/tensorflow/tflite-micro/actions/workflows/generate_integration_tests.yml) |


# Contributing
See our [contribution documentation](CONTRIBUTING.md).

# Getting Help

A [Github issue](https://github.com/tensorflow/tflite-micro/issues/new/choose)
should be the primary method of getting in touch with the TensorFlow Lite Micro
(TFLM) team.

The following resources may also be useful:

1.  SIG Micro [email group](https://groups.google.com/a/tensorflow.org/g/micro)
    and
    [monthly meetings](http://doc/1YHq9rmhrOUdcZnrEnVCWvd87s2wQbq4z17HbeRl-DBc).

1.  SIG Micro [gitter chat room](https://gitter.im/tensorflow/sig-micro).

1. For questions that are not specific to TFLM, please consult the broader TensorFlow project, e.g.:
   * Create a topic on the [TensorFlow Discourse forum](https://discuss.tensorflow.org)
   * Send an email to the [TensorFlow Lite mailing list](https://groups.google.com/a/tensorflow.org/g/tflite)
   * Create a [TensorFlow issue](https://github.com/tensorflow/tensorflow/issues/new/choose)
   * Create a [Model Optimization Toolkit](https://github.com/tensorflow/model-optimization) issue

# Additional Documentation

 * [Continuous Integration](docs/continuous_integration.md)
 * [Benchmarks](tensorflow/lite/micro/benchmarks/README.md)
 * [Profiling](tensorflow/lite/micro/docs/profiling.md)
 * [Memory Management](tensorflow/lite/micro/docs/memory_management.md)
 * [Logging](tensorflow/lite/micro/docs/logging.md)
 * [Porting Reference Kernels from TfLite to TFLM](tensorflow/lite/micro/docs/porting_reference_ops.md)
 * [Optimized Kernel Implementations](tensorflow/lite/micro/docs/optimized_kernel_implementations.md)
 * [New Platform Support](tensorflow/lite/micro/docs/new_platform_support.md)
 * Platform/IP support
   * [Arm IP support](tensorflow/lite/micro/docs/arm.md)
 * [Software Emulation with Renode](tensorflow/lite/micro/docs/renode.md)
 * [Software Emulation with QEMU](tensorflow/lite/micro/docs/qemu.md)
 * [Python Dev Guide](docs/python.md)
 * [Automatically Generated Files](docs/automatically_generated_files.md)
 * [Python Interpreter Guide](python/tflite_micro/README.md)

# RFCs

1. [Pre-allocated tensors](tensorflow/lite/micro/docs/rfc/001_preallocated_tensors.md)
1. [TensorFlow Lite for Microcontrollers Port of 16x8 Quantized Operators](tensorflow/lite/micro/docs/rfc/002_16x8_quantization_port.md)
