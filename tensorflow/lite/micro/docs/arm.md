<!--ts-->
* [Arm(R) IP support in Tensorflow Lite for Microcontrollers (TFLM)](#arm-ip)
   * [Arm(R) Cortex(R)-M processor family](#cortex-m)
   * [CMSIS-NN optimized library](#cmsis-nn)
   * [Arm(R) Ethos(TM)-U microNPU family](#ethos-u)
   * [Arm(R) Corstone(TM)-300 FVP](#corstone-300)
<!--te-->

# Arm(R) IP support in Tensorflow Lite for Microcontrollers (TFLM)

This doc outlines how to use Arm IP with TFLM. The following sub chapters
contain more details of the respective IP.

## Arm(R) Cortex(R)-M processor family
Arm's Cortex-M processor support is fully integrated to TFLM. To build a TFLM
library for any Cortex-M processor, check out the [Cortex-M generic readme](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/cortex_m_generic/README.md).
Additionally, CMSIS-NN provides optimal performance executing machine learning
workloads on Cortex-M. See the [sub chapter CMSIS-NN](#cmsis-nn).


## CMSIS-NN optimized library
Common Microcontroller Software Interface Standard for Neural Networks
(CMSIS-NN) is a collection of efficient neural network kernels developed to
maximize performance on Cortex-M processors. The CMSIS-NN optimized kernel are
highly integrated to TFLM. For more information how to utilize these kernels,
see [CMSIS-NN readme](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/kernels/cmsis_nn/README.md).


## Arm(R) Ethos(TM)-U microNPU family
The Ethos-U microNPU (Neural Processing Unit) family consist of [Ethos-U55](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u55)
and [Ethos-U65](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u65).
Ethos-U55 is designed to accelerate ML inference in area-constrained embedded
and IoT devices, whereas Ethos-U65 extends its applicability to be used as an
Cortex-M subsystem to a larger Arm Cortex-A, Cortex-R and Neoverse-based system.

To get started with TFLM and Ethos-U, see the [Ethos-U readme](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/kernels/ethos_u/README.md).


## Arm(R) Corstone(TM)-300 FVP
[Corstone-300](https://developer.arm.com/Processors/Corstone-300) is a hardware
reference design based on the Arm Cortex-M55 processor, which integrates the
Ethos-U55 microNPU. The [Corstone-300 FVP](https://developer.arm.com/tools-and-software/open-source-software/arm-platforms-software/arm-ecosystem-fvps)
(Fixed Virtual Platform) is a model of the hardware which enables execution of
full software stacks ahead of silicon.

To get started with TFLM and Corstone-300 FVP, see the [Corstone-300 readme](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/cortex_m_corstone_300/README.md).

