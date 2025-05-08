# CMSIS-NN {#mainpage}

**CMSIS-NN** is an open-source software library that provides a collection of efficient neural network (NN) kernels developed to maximize the performance and minimize the memory footprint of neural networks running on Arm Cortex-M processors.

![Overview of CMSIS-NN](./images/cmsis-nn-overview.png)

CMSIS-NN functions are implemented in several variants and the optimal solution is automatically taken at compile time depending on the features available on the target processor architecture. Implementations for the following features are available:

 - Single Instruction Multiple Data (SIMD) capability (e.g, Cortex-M0)
 - DSP extension (e.g Cortex-M4)
 - Arm M-Profile Vector Extension(MVE) (e.g Cortex-M55).

## Access to CMSIS-NN

CMSIS-NN is actively maintained in a GitHub repository and is released as a standalone package in CMSIS-Pack format.

 - [**CMSIS-NN GitHub Repo**](https://github.com/ARM-software/CMSIS-NN) provides the full source code of CMSIS-NN kernels.
 - [**CMSIS-NN Documentation**](https://arm-software.github.io/CMSIS-NN/latest/) explains how to use the library and describes the implemented functions in details.
 - [**CMSIS-NN Pack**](https://www.keil.arm.com/packs/cmsis-nn-arm/versions/) delivers CMSIS-NN components and examples in [CMSIS-Pack format](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/index.html).

## Key Features and Benefits

 - CMSIS-NN provides a broad set of neural network kernels for Cortex-M devices.
 - Optimized implementations for different Cortex-M capabilities (SIMD, FPU, MVE).
 - Arm Compiler 6 and on Arm GNU Toolchain support.
 - Follows int8 and int16 quantization specification of TensorFlow Lite for Microcontrollers.
 - Widely adopted in the industry.
