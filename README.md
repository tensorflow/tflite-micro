# TensorFlow Lite for Microcontrollers (TFLM)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Core (Linux)](https://github.com/tensorflow/tflite-micro/actions/workflows/run_core.yml/badge.svg?branch=main)](https://github.com/tensorflow/tflite-micro/actions/workflows/run_core.yml)
[![Windows](https://github.com/tensorflow/tflite-micro/actions/workflows/run_windows.yml/badge.svg?branch=main)](https://github.com/tensorflow/tflite-micro/actions/workflows/run_windows.yml)
[![Cortex-M](https://github.com/tensorflow/tflite-micro/actions/workflows/run_cortex_m.yml/badge.svg?branch=main)](https://github.com/tensorflow/tflite-micro/actions/workflows/run_cortex_m.yml)
[![Xtensa](https://github.com/tensorflow/tflite-micro/actions/workflows/run_xtensa.yml/badge.svg?branch=main)](https://github.com/tensorflow/tflite-micro/actions/workflows/run_xtensa.yml)
[![RISC-V](https://github.com/tensorflow/tflite-micro/actions/workflows/run_riscv.yml/badge.svg?branch=main)](https://github.com/tensorflow/tflite-micro/actions/workflows/run_riscv.yml)
[![Hexagon](https://github.com/tensorflow/tflite-micro/actions/workflows/run_hexagon.yml/badge.svg?branch=main)](https://github.com/tensorflow/tflite-micro/actions/workflows/run_hexagon.yml)
[![PyPI](https://github.com/tensorflow/tflite-micro/actions/workflows/pypi_build.yml/badge.svg?branch=main)](https://github.com/tensorflow/tflite-micro/actions/workflows/pypi_build.yml)

TensorFlow Lite for Microcontrollers is designed to run deep learning models on bare-metal systems, DSPs, and microcontrollers with limited memory (tens to hundreds of kilobytes). It runs models converted via [LiteRT](https://github.com/google-ai-edge/LiteRT) and [TensorFlow](https://github.com/tensorflow/tensorflow/).

---

## Supported Hardware & Ecosystems

* **Tier-1 Architectures (Tested in CI):** Arm Cortex-M (CMSIS-NN / Corstone), Cadence Xtensa (HiFi 3/4/5, Vision P6), RISC-V, Qualcomm Hexagon, and x86/Windows.
* **Community Platforms & SDKs:** [Arduino](https://github.com/tensorflow/tflite-micro-arduino-examples), [Coral Dev Board Micro](https://github.com/google-coral/coralmicro), [Espressif ESP32](https://github.com/espressif/tflite-micro-esp-examples), [Renesas](https://github.com/renesas/tflite-micro-renesas), [Silicon Labs](https://github.com/SiliconLabsSoftware/machine_learning_applications/), and [Ingenic MIPS](https://github.com/yinzara/ingenic-tflite-micro).

---

## Documentation

* **Core Concepts:**
  * [Memory Management](tensorflow/lite/micro/docs/memory_management.md)
  * [Error Handling & Defensive Programming](tensorflow/lite/micro/docs/error_handling_guide.md)
  * [Logging](tensorflow/lite/micro/docs/logging.md)
  * [Continuous Integration](docs/continuous_integration.md)
* **Porting & Hardware Optimization:**
  * [Optimized Kernel Implementations](tensorflow/lite/micro/docs/optimized_kernel_implementations.md)
  * [New Platform Support](tensorflow/lite/micro/docs/new_platform_support.md)
  * [Arm IP Support](tensorflow/lite/micro/docs/arm.md)
  * [Software Emulation (Renode)](tensorflow/lite/micro/docs/renode.md) and [QEMU](tensorflow/lite/micro/docs/qemu.md)
* **Tooling, Compression & Python:**
  * [Model Compression Guide](tensorflow/lite/micro/docs/compression.md) & [Tutorial](tensorflow/lite/micro/compression/tutorial/mnist_compression_tutorial.md)
  * [Python Interpreter & Dev Guide](python/tflite_micro/README.md)
  * [Benchmarks](tensorflow/lite/micro/benchmarks/README.md) and [Profiling](tensorflow/lite/micro/docs/profiling.md)

---

## Community & Contributing

* **Questions & Issues:** Please file a [GitHub Issue](https://github.com/tensorflow/tflite-micro/issues/new/choose).
* **Contributing:** Review our [Contributing Guide](CONTRIBUTING.md).
* **Security:** Report vulnerabilities according to our [Security Policy](SECURITY.md).
